"""Read-only adoption of the failed AIDS c766 adaptive-anchor selection.

The c766 run is *not* a successful DBSCAN result.  Its adaptive seed/failure
scan completed and published a deterministic 266-row selection before the
exact anchor graph proved disconnected.  A later recovery route may reuse
only that failed-selection evidence.  This module therefore publishes a
separate typed receipt whose recovery-only READY marker means "the failed
evidence was reopened exactly".  It deliberately never creates a file named
``PASS`` and can never satisfy an ordinary scientific dependency.

Production entry points are pinned to one AutoDL control namespace, controller
ID, and byte identities.  Paths to gates, states, and task outputs are derived
from that authority instead of accepted from the command line.  No source
artifact is copied, linked, rewritten, or signalled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from src.baselines.comrecgc.aids_pair_semantics import (
    AIDS_PAIR_SEMANTICS_SCHEMA,
)
from src.baselines.comrecgc.close_pair_view import (
    ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA,
    CLOSE_PAIR_VIEW_SCHEMA,
    FILTER_OPERATOR,
    NORMALIZED_DISTANCE_CONTRACT,
    PAIR_ORDER,
    PAIR_ORIENTATION,
    SCALE_CONTRACT,
)
from src.baselines.comrecgc.external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ADAPTIVE_SELECTION_SCHEMA_VERSION,
    SCHEMA_VERSION as DBSCAN_SCHEMA_VERSION,
    _load_checkpoint as _load_dbscan_checkpoint,
    _load_progress_ledgers,
    _sample_indices_sha256,
    _stable_hash as _dbscan_stable_hash,
    _validate_adaptive_selection_manifest,
)
from src.baselines.comrecgc.external_memory_recourse import PAIR_STORE_SCHEMA


ADOPTION_SCHEMA_VERSION = "aids_comrecgc_c766_failed_selection_adoption_v3"
RECEIPT_NAME = "failed_selection_adoption_receipt.json"
PRETERMINAL_NAME = ".failed_selection_adoption_receipt.preterminal.json"
READY_NAME = "RECOVERY_EVIDENCE_READY"
READY_PREPARED_NAME = ".RECOVERY_EVIDENCE_READY.prepared"
READY_PREFIX = b"AIDS_C766_FAILED_SELECTION_RECOVERY_EVIDENCE_READY_V3\n"
MUTABLE_STATE_VALUE = "<MUTABLE>"
PAIR_BITMAP_COMPARE_BLOCK_ROWS = 1_000_000

_SAFE_WORKER_EXIT_OBSERVATIONS = frozenset(
    {
        "ORIGINAL_GENERATION_EXITED_PID_ABSENT",
        "ORIGINAL_GENERATION_EXITED_PID_REUSED",
        "ORIGINAL_GENERATION_EXITED_ZOMBIE",
    }
)
_SAFE_RECORDED_CHILD_EXIT_OBSERVATIONS = frozenset(
    {
        "RECORDED_CHILD_PID_ABSENT",
        "RECORDED_CHILD_ZOMBIE",
    }
)
_NO_RECORDED_CHILD_OBSERVATION = "NO_RECORDED_CHILD_PID"
_PROCESS_EXIT_KEYS = frozenset(
    {
        "expected_worker_identity",
        "worker_observation",
        "recorded_child_pid",
        "child_observation",
        "old_science_worker_exited",
        "signals_sent",
    }
)

TASK_STATE_TOP_KEYS = frozenset(
    {
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
)
TASK_STATE_MAIN_KEYS = frozenset(
    {
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
)
TASK_STATE_IDENTITY_KEYS = frozenset({"command_sha256", "pid", "start_ticks"})

PRODUCTION_CONTROL_ROOT = Path(
    "/autodl-fs/data/counterfactual-subgraph-runtime/control"
)
PRODUCTION_OUTPUT_PARENT = Path(
    "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
    "recovery_evidence/aids_c766_failed_selection_v1"
)
PRODUCTION_PROC_ROOT = Path("/proc")
CONTROL_NAMESPACE = "four_methods_four_datasets_continuation"
SOURCE_CONTROLLER_ID = (
    "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1"
)
CLOSE_TASK_ID = "aids_comrecgc_theta_close_view_v1"
FINAL_TASK_ID = "aids_comrecgc_standardized_exact_route_v5_snapshot_adopt_v1"
FAILED_TREE_RELATIVE_PATHS = (
    "FAILED.json",
    "common_recourse/external_memory/dbscan/adaptive_anchor_selection.json",
    "common_recourse/external_memory/dbscan/adaptive_first_pass_failure_indices.npy",
    "common_recourse/external_memory/dbscan/adaptive_selected_anchor_rows.npy",
    "common_recourse/external_memory/dbscan/checkpoint.json",
    "common_recourse/external_memory/dbscan/shortcut_anchor_edges.npy",
    "common_recourse/external_memory/dbscan/shortcut_anchor_indices.npy",
    "common_recourse/external_memory/dbscan/shortcut_failure.json",
    "common_recourse/external_memory/pair_store_adoption/run_manifest.json",
    "continuation_resume_contract.json",
    "generation_adoption_manifest.json",
    "stage_checkpoints/common_recourse.json",
    "stage_state.json",
    "upstream_checkout_audit.json",
)


class FailedSelectionAdoptionError(RuntimeError):
    """The failed-selection evidence is not safe to adopt."""


@dataclass(frozen=True)
class TaskStateAuthority:
    """Canonical non-heartbeat projection of one persistent task state."""

    schema_version: int
    dataset: str
    stage: str
    task_id: str
    state: str
    reason: str | None
    instance_id: str
    attempt: int
    run_id: str
    expected_output: str
    command_sha256: str
    child_pid: int
    failure_class: str | None
    launcher_pid: int
    launcher_start_ticks: int
    launcher_command_sha256: str
    worker_pid: int
    worker_start_ticks: int
    worker_command_sha256: str
    failure_reason_length: int | None
    failure_reason_sha256: str | None
    projection_sha256: str
    created_at: str = "2026-08-24T19:14:15+00:00"
    adopted: bool = False
    gpu_colocation_gate: str | None = None
    gpu_colocation_gate_sha256: str | None = None
    gpu_index: int | None = None
    gpu_lock_mode: str = "exclusive"
    gpu_memory_reservation_mb: int = 0
    gpu_shared_workload_class: str | None = None
    gpu_uuid: str | None = None
    log_path: str = "/fixture/controller.log"
    oom_retry_count: int = 0
    required_absolute_output_files: tuple[str, ...] = ()
    resume_from_checkpoint: str | None = None
    resume_source_output: str | None = None
    retry_kind: str | None = None
    started_at: str = "2026-08-24T22:48:39+00:00"
    tmux_session: str | None = None
    transient_retry_count: int = 0

    def projection(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "stage": self.stage,
            "task_id": self.task_id,
            "state": self.state,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": MUTABLE_STATE_VALUE,
            "instances": {
                "main": {
                    "instance_id": self.instance_id,
                    "state": self.state,
                    "attempt": self.attempt,
                    "run_id": self.run_id,
                    "adopted": self.adopted,
                    "expected_output": self.expected_output,
                    "command_sha256": self.command_sha256,
                    "child_pid": self.child_pid,
                    "failure_class": self.failure_class,
                    "launcher_pid": self.launcher_pid,
                    "launcher_identity": {
                        "pid": self.launcher_pid,
                        "start_ticks": self.launcher_start_ticks,
                        "command_sha256": self.launcher_command_sha256,
                    },
                    "worker_pid": self.worker_pid,
                    "worker_identity": {
                        "pid": self.worker_pid,
                        "start_ticks": self.worker_start_ticks,
                        "command_sha256": self.worker_command_sha256,
                    },
                    "gpu_index": self.gpu_index,
                    "gpu_uuid": self.gpu_uuid,
                    "gpu_lock_mode": self.gpu_lock_mode,
                    "gpu_memory_reservation_mb": self.gpu_memory_reservation_mb,
                    "gpu_shared_workload_class": self.gpu_shared_workload_class,
                    "gpu_colocation_gate": self.gpu_colocation_gate,
                    "gpu_colocation_gate_sha256": self.gpu_colocation_gate_sha256,
                    "log_path": self.log_path,
                    "required_absolute_output_files": list(
                        self.required_absolute_output_files
                    ),
                    "started_at": self.started_at,
                    "tmux_session": self.tmux_session,
                    "heartbeat_at": MUTABLE_STATE_VALUE,
                    "oom_retry_count": self.oom_retry_count,
                    "transient_retry_count": self.transient_retry_count,
                    "retry_kind": self.retry_kind,
                    "resume_from_checkpoint": self.resume_from_checkpoint,
                    "resume_source_output": self.resume_source_output,
                    "failure_reason": (
                        None
                        if self.failure_reason_length is None
                        else {
                            "length": self.failure_reason_length,
                            "sha256": self.failure_reason_sha256,
                        }
                    ),
                }
            },
        }

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.instance_id != "main"
            or type(self.attempt) is not int
            or self.attempt < 0
            or not Path(self.expected_output).is_absolute()
            or not isinstance(self.created_at, str)
            or not self.created_at
            or type(self.adopted) is not bool
            or not isinstance(self.gpu_lock_mode, str)
            or not self.gpu_lock_mode
            or type(self.gpu_memory_reservation_mb) is not int
            or self.gpu_memory_reservation_mb < 0
            or not isinstance(self.log_path, str)
            or not self.log_path
            or not isinstance(self.started_at, str)
            or not self.started_at
            or type(self.oom_retry_count) is not int
            or self.oom_retry_count < 0
            or type(self.transient_retry_count) is not int
            or self.transient_retry_count < 0
            or not isinstance(self.required_absolute_output_files, tuple)
            or any(
                not isinstance(value, str) or not Path(value).is_absolute()
                for value in self.required_absolute_output_files
            )
            or not _is_sha256(self.command_sha256)
            or not _is_sha256(self.launcher_command_sha256)
            or not _is_sha256(self.worker_command_sha256)
            or not _is_sha256(self.projection_sha256)
            or not _is_plain_positive_int(self.launcher_pid)
            or not _is_plain_positive_int(self.worker_pid)
            or not _is_plain_positive_int(self.child_pid)
            or self.child_pid == self.worker_pid
            or self.launcher_pid != self.worker_pid
            or self.launcher_start_ticks != self.worker_start_ticks
            or self.launcher_command_sha256 != self.worker_command_sha256
            or not _is_plain_positive_int(self.launcher_start_ticks)
            or not _is_plain_positive_int(self.worker_start_ticks)
            or (self.failure_reason_length is None)
            != (self.failure_reason_sha256 is None)
            or (
                self.failure_reason_length is not None
                and (
                    type(self.failure_reason_length) is not int
                    or self.failure_reason_length < 0
                )
            )
            or (
                self.failure_reason_sha256 is not None
                and not _is_sha256(self.failure_reason_sha256)
            )
            or _stable_hash(self.projection()) != self.projection_sha256
        ):
            raise FailedSelectionAdoptionError(
                f"task-state projection authority changed: {self.task_id}"
            )


@dataclass(frozen=True)
class FailedSelectionAuthority:
    """All mutable-looking values that production freezes before adoption.

    Tests use a private fixture profile with small arrays.  Public production
    wrappers below always use :data:`PRODUCTION_AUTHORITY` and reject alternate
    control/proc roots.
    """

    control_root: str
    output_parent: str
    proc_root: str
    namespace: str
    controller_id: str
    close_task_id: str
    final_task_id: str
    close_state_authority: TaskStateAuthority
    final_state_authority: TaskStateAuthority
    failed_tree_files: tuple[tuple[str, str], ...]
    controller_manifest_sha256: str
    close_gate_sha256: str
    close_manifest_sha256: str
    final_gate_sha256: str
    checkpoint_sha256: str
    shortcut_failure_sha256: str
    selection_manifest_sha256: str
    failure_indices_sha256: str
    anchor_indices_sha256: str
    anchor_rows_sha256: str
    anchor_edges_sha256: str
    physical_pairs_sha256: str
    physical_vectors_sha256: str
    physical_rows: int
    vector_features: int
    parent_count: int
    candidate_count: int
    failure_count: int
    anchor_count: int
    anchor_edge_count: int
    seed_count: int
    initial_component_sizes: tuple[int, ...]
    eps: float = 0.02
    min_samples: int = 3
    theta: float = 0.1

    def validate(self) -> None:
        self.close_state_authority.validate()
        self.final_state_authority.validate()
        failed_tree = dict(self.failed_tree_files)
        if (
            len(failed_tree) != len(self.failed_tree_files)
            or not failed_tree
            or set(failed_tree) != set(FAILED_TREE_RELATIVE_PATHS)
            or any(
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not _is_sha256(digest)
                for relative, digest in self.failed_tree_files
            )
        ):
            raise FailedSelectionAdoptionError(
                "failed-tree production allowlist is invalid"
            )
        expected_selection_hashes = {
            "common_recourse/external_memory/dbscan/adaptive_anchor_selection.json": (
                self.selection_manifest_sha256
            ),
            "common_recourse/external_memory/dbscan/adaptive_first_pass_failure_indices.npy": (
                self.failure_indices_sha256
            ),
            "common_recourse/external_memory/dbscan/adaptive_selected_anchor_rows.npy": (
                self.anchor_rows_sha256
            ),
            "common_recourse/external_memory/dbscan/checkpoint.json": (
                self.checkpoint_sha256
            ),
            "common_recourse/external_memory/dbscan/shortcut_anchor_edges.npy": (
                self.anchor_edges_sha256
            ),
            "common_recourse/external_memory/dbscan/shortcut_anchor_indices.npy": (
                self.anchor_indices_sha256
            ),
            "common_recourse/external_memory/dbscan/shortcut_failure.json": (
                self.shortcut_failure_sha256
            ),
        }
        if any(
            failed_tree.get(relative) != digest
            for relative, digest in expected_selection_hashes.items()
        ):
            raise FailedSelectionAdoptionError(
                "failed-tree selection hashes disagree with the authority profile"
            )
        if (
            self.close_state_authority.task_id != self.close_task_id
            or self.close_state_authority.state != "PASS"
            or self.close_state_authority.reason is not None
            or self.close_state_authority.failure_class is not None
            or self.final_state_authority.task_id != self.final_task_id
            or self.final_state_authority.state != "FAILED"
            or self.final_state_authority.reason != "main:SEMANTIC"
            or self.final_state_authority.failure_class != "SEMANTIC"
            or self.final_state_authority.failure_reason_length is None
            or self.close_state_authority.expected_output
            == self.final_state_authority.expected_output
        ):
            raise FailedSelectionAdoptionError(
                "close/final task-state authorities changed"
            )
        if not Path(self.control_root).is_absolute():
            raise FailedSelectionAdoptionError("authority control_root is not absolute")
        if not Path(self.output_parent).is_absolute():
            raise FailedSelectionAdoptionError("authority output_parent is not absolute")
        if not Path(self.proc_root).is_absolute():
            raise FailedSelectionAdoptionError("authority proc_root is not absolute")
        for name, value in asdict(self).items():
            if name.endswith("_sha256") and not _is_sha256(value):
                raise FailedSelectionAdoptionError(
                    f"authority has an invalid SHA256: {name}"
                )
        if (
            self.physical_rows
            != self.parent_count * self.candidate_count
            or self.physical_rows <= 0
            or self.vector_features <= 0
            or self.failure_count < 0
            or self.anchor_count != self.failure_count + self.seed_count
            or sum(self.initial_component_sizes) != self.anchor_count
            or self.anchor_edge_count < 0
            or self.min_samples != 3
            or self.eps != 0.02
            or self.theta != 0.1
        ):
            raise FailedSelectionAdoptionError("authority dimensions/protocol changed")


PRODUCTION_AUTHORITY = FailedSelectionAuthority(
    control_root=str(PRODUCTION_CONTROL_ROOT),
    output_parent=str(PRODUCTION_OUTPUT_PARENT),
    proc_root=str(PRODUCTION_PROC_ROOT),
    namespace=CONTROL_NAMESPACE,
    controller_id=SOURCE_CONTROLLER_ID,
    close_task_id=CLOSE_TASK_ID,
    final_task_id=FINAL_TASK_ID,
    close_state_authority=TaskStateAuthority(
        schema_version=1,
        dataset="aids",
        stage="AM_COMRECGC_HELDOUT_EVAL",
        task_id=CLOSE_TASK_ID,
        state="PASS",
        reason=None,
        instance_id="main",
        attempt=0,
        run_id=(
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_"
            "snapshot_adopt_v1-aids_comrecgc_theta_close_view_v1-main-a0"
        ),
        expected_output=(
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
            "repairs/four_methods_four_datasets_aids_comrecgc_exact_route_v5_"
            "snapshot_adopt_v1/close_pair_view/attempt-0"
        ),
        command_sha256=(
            "db6913cf88941bd7ea67b91eb1053b17e6d5f64aefc1a0e9f058e38d29fa434a"
        ),
        child_pid=584_468,
        failure_class=None,
        launcher_pid=584_467,
        launcher_start_ticks=704_335_146,
        launcher_command_sha256=(
            "a0937597683c5ba74b590472ae0de27c1c6a77e968031d824e95626c610f8480"
        ),
        worker_pid=584_467,
        worker_start_ticks=704_335_146,
        worker_command_sha256=(
            "a0937597683c5ba74b590472ae0de27c1c6a77e968031d824e95626c610f8480"
        ),
        failure_reason_length=None,
        failure_reason_sha256=None,
        projection_sha256=(
            "f2bcde0b4cf8b86082abb3bc9b7499c8a9459f1a1df92d8eada28996e332a780"
        ),
        created_at="2026-08-24T19:14:15+00:00",
        log_path=(
            "/autodl-fs/data/counterfactual-subgraph-runtime/logs/"
            "four_methods_four_datasets_continuation/"
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1/"
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1-"
            "aids_comrecgc_theta_close_view_v1-main-a0.log"
        ),
        started_at="2026-08-24T22:00:03+00:00",
    ),
    final_state_authority=TaskStateAuthority(
        schema_version=1,
        dataset="aids",
        stage="AM_COMRECGC_HELDOUT_EVAL",
        task_id=FINAL_TASK_ID,
        state="FAILED",
        reason="main:SEMANTIC",
        instance_id="main",
        attempt=0,
        run_id=(
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_"
            "snapshot_adopt_v1-aids_comrecgc_standardized_exact_route_v5_snap"
        ),
        expected_output=(
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/"
            "repairs/four_methods_four_datasets_aids_comrecgc_exact_route_v5_"
            "snapshot_adopt_v1/cells/aids/comrecgc/standardized/attempt-0"
        ),
        command_sha256=(
            "195419680412560b143cfd58d6f2e2e85edd837fe567aad6f8099b2b274c553c"
        ),
        child_pid=590_228,
        failure_class="SEMANTIC",
        launcher_pid=590_227,
        launcher_start_ticks=704_626_812,
        launcher_command_sha256=(
            "1f582f067371ad93d44975b6e97d162cf2acedee35c9b5cf6a11b1788a70683e"
        ),
        worker_pid=590_227,
        worker_start_ticks=704_626_812,
        worker_command_sha256=(
            "1f582f067371ad93d44975b6e97d162cf2acedee35c9b5cf6a11b1788a70683e"
        ),
        failure_reason_length=1_837,
        failure_reason_sha256=(
            "dec892bdb547a2f085866c4cbb76bc45210fd22795f635684aca040212627bd7"
        ),
        projection_sha256=(
            "b455b618d29ac807eecead64b3aa8f47bfdee67344dab9cfb566337d148c12ab"
        ),
        created_at="2026-08-24T19:14:15+00:00",
        log_path=(
            "/autodl-fs/data/counterfactual-subgraph-runtime/logs/"
            "four_methods_four_datasets_continuation/"
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1/"
            "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1-"
            "aids_comrecgc_standardized_exact_route_v5_snap.log"
        ),
        started_at="2026-08-24T22:48:39+00:00",
    ),
    failed_tree_files=(
        (
            "FAILED.json",
            "0562909dc05c5bc4f3ae75235e1f3f17c4579558527969ff27e900ad96779b71",
        ),
        (
            "common_recourse/external_memory/dbscan/adaptive_anchor_selection.json",
            "0c3e569d65fa299e2658321bf5cfd2961c0cab1d19cd49aa2799045d1cab6e8e",
        ),
        (
            "common_recourse/external_memory/dbscan/"
            "adaptive_first_pass_failure_indices.npy",
            "b56883c3c79d60e6cd582eb071278b78289b63a0b724ab55a407cd506d0502be",
        ),
        (
            "common_recourse/external_memory/dbscan/adaptive_selected_anchor_rows.npy",
            "ff32eec327569527862cf18d1d9dbe5ac374a63486e757cb1e5587494a976012",
        ),
        (
            "common_recourse/external_memory/dbscan/checkpoint.json",
            "fa3ed4a566f1518876ebc58e3bbd0fc1e87d8d6b2a44a53576fba398f1fd0a3b",
        ),
        (
            "common_recourse/external_memory/dbscan/shortcut_anchor_edges.npy",
            "91aacf23b644ed89e11247b4e6c23ffe6b7c7cc994b3ab9bf0ef8e10e78f2f3a",
        ),
        (
            "common_recourse/external_memory/dbscan/shortcut_anchor_indices.npy",
            "0f70593d7d632fd1040ea5c5fbf128552afbb917185dd673832cd9214db028c4",
        ),
        (
            "common_recourse/external_memory/dbscan/shortcut_failure.json",
            "74bc3d73e99568b2cc05dfda3d62d39538acc1dae9ccd9fb4743346ad1e6cea5",
        ),
        (
            "common_recourse/external_memory/pair_store_adoption/run_manifest.json",
            "d26d4817078bcbd0e10690031f096ba728b6bd248ebaec5b147d4725f17b29fd",
        ),
        (
            "continuation_resume_contract.json",
            "364d7ffee290b7f24f16ab4b3cbb9c0bfe73a7e71ca3b74ea48ba637060f43a6",
        ),
        (
            "generation_adoption_manifest.json",
            "09437f88f407760b68906a1295d49243c6a74686f5dbe8b0905a1a2f6ac1b8c8",
        ),
        (
            "stage_checkpoints/common_recourse.json",
            "878190dfbb1a5883b91ec9f76bb8d03d1e44c6c8df697a9ed9ad29fe8dc3868e",
        ),
        (
            "stage_state.json",
            "878190dfbb1a5883b91ec9f76bb8d03d1e44c6c8df697a9ed9ad29fe8dc3868e",
        ),
        (
            "upstream_checkout_audit.json",
            "88942aed463c02509c5f6093f56a4e7623d287c1ee12802bf70def400f098b4a",
        ),
    ),
    controller_manifest_sha256=(
        "7b2987bc2d223ebe3262cc15bc43bd1c0b030c6706a1c074959d154af5fd84d7"
    ),
    close_gate_sha256=(
        "042837003d8e07c41d10c283909f5dc545659d6a2ad99db25d8652509ac03e8b"
    ),
    close_manifest_sha256=(
        "d41792e65bc9989c9a2c0abb9ef4c552ed863c9e362d9bba72cf1cc6dd5d331a"
    ),
    final_gate_sha256=(
        "a7c46f485a18a42d5dce081528945cc859c5c53d4ad7a343c62ecb246089e65b"
    ),
    checkpoint_sha256=(
        "fa3ed4a566f1518876ebc58e3bbd0fc1e87d8d6b2a44a53576fba398f1fd0a3b"
    ),
    shortcut_failure_sha256=(
        "74bc3d73e99568b2cc05dfda3d62d39538acc1dae9ccd9fb4743346ad1e6cea5"
    ),
    selection_manifest_sha256=(
        "0c3e569d65fa299e2658321bf5cfd2961c0cab1d19cd49aa2799045d1cab6e8e"
    ),
    failure_indices_sha256=(
        "b56883c3c79d60e6cd582eb071278b78289b63a0b724ab55a407cd506d0502be"
    ),
    anchor_indices_sha256=(
        "0f70593d7d632fd1040ea5c5fbf128552afbb917185dd673832cd9214db028c4"
    ),
    anchor_rows_sha256=(
        "ff32eec327569527862cf18d1d9dbe5ac374a63486e757cb1e5587494a976012"
    ),
    anchor_edges_sha256=(
        "91aacf23b644ed89e11247b4e6c23ffe6b7c7cc994b3ab9bf0ef8e10e78f2f3a"
    ),
    physical_pairs_sha256=(
        "c83eba699f2b269bc92ab6b1be434c77a16d4f4113085150de6704b9f1a1df57"
    ),
    physical_vectors_sha256=(
        "68072364166c20364b8d079a08fd67f5008447db54f51b338f3f541eb54b39e5"
    ),
    physical_rows=91_916_686,
    vector_features=64,
    parent_count=1_283,
    candidate_count=71_642,
    failure_count=263,
    anchor_count=266,
    anchor_edge_count=16_860,
    seed_count=3,
    initial_component_sizes=(114, 149, 3),
)


@dataclass(frozen=True)
class _AuthorityPaths:
    control_root: Path
    namespace_root: Path
    source_manifest: Path
    controller_root: Path
    controller_snapshot: Path
    close_gate: Path
    close_state: Path
    final_gate: Path
    final_state: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(
        parsed
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _is_plain_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _ready_marker_bytes(receipt_bytes: bytes) -> bytes:
    return (
        READY_PREFIX
        + b"receipt_sha256="
        + hashlib.sha256(receipt_bytes).hexdigest().encode("ascii")
        + b"\n"
    )


def _profile_payload(profile: FailedSelectionAuthority) -> dict[str, Any]:
    """Return the JSON-domain authority representation used by receipts."""

    value = json.loads(json.dumps(asdict(profile), sort_keys=True))
    assert isinstance(value, dict)
    return value


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "nlink": int(value.st_nlink),
    }


def _directory_inode_identity(value: os.stat_result) -> dict[str, int]:
    """Return only directory attributes that cannot change on content churn."""

    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
    }


class _HeldDirectorySet:
    """Keep named authority directories open and inode-bound during a scan."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[int, dict[str, int], str]] = {}

    def add(self, path: Path, *, label: str) -> Path:
        physical = _physical_dir(path, label=label)
        existing = self._entries.get(physical)
        if existing is not None:
            self.assert_one(physical)
            return physical
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(physical, flags)
        except OSError as exc:
            raise FailedSelectionAdoptionError(
                f"cannot hold {label} without following"
            ) from exc
        opened = os.fstat(descriptor)
        named = physical.lstat()
        identity = _directory_inode_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or identity != _directory_inode_identity(named)
        ):
            os.close(descriptor)
            raise FailedSelectionAdoptionError(f"{label} inode changed while opening")
        self._entries[physical] = (descriptor, identity, label)
        return physical

    def assert_one(self, path: Path, *, named: bool = True) -> None:
        descriptor, identity, label = self._entries[path]
        if _directory_inode_identity(os.fstat(descriptor)) != identity:
            raise FailedSelectionAdoptionError(f"held {label} inode changed")
        if named:
            try:
                current = path.lstat()
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    f"named {label} disappeared"
                ) from exc
            if (
                not stat.S_ISDIR(current.st_mode)
                or _directory_inode_identity(current) != identity
            ):
                raise FailedSelectionAdoptionError(f"named {label} inode changed")

    def assert_all(self, *, named: bool = True) -> None:
        for path in sorted(self._entries, key=str):
            self.assert_one(path, named=named)

    def descriptor(self, path: Path) -> int:
        physical = Path(path)
        if physical not in self._entries:
            raise FailedSelectionAdoptionError(
                f"authority directory was not held: {physical}"
            )
        self.assert_one(physical)
        return self._entries[physical][0]

    def evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(path),
                "identity": dict(identity),
                "label": label,
                "opened_with_o_directory_o_nofollow": True,
            }
            for path, (_descriptor, identity, label) in sorted(
                self._entries.items(), key=lambda row: str(row[0])
            )
        ]

    def close(self) -> None:
        for descriptor, _identity, _label in self._entries.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._entries.clear()

    def __enter__(self) -> "_HeldDirectorySet":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _dbscan_stat_identity(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: int(value[key])
        for key in ("device", "inode", "mode", "size", "mtime_ns", "ctime_ns")
    }


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise FailedSelectionAdoptionError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            value = current.lstat()
        except FileNotFoundError:
            raise FailedSelectionAdoptionError(f"{label} is absent: {current}")
        except OSError as exc:
            raise FailedSelectionAdoptionError(
                f"cannot inspect {label}: {current}"
            ) from exc
        if stat.S_ISLNK(value.st_mode):
            raise FailedSelectionAdoptionError(
                f"{label} contains a symlink component: {current}"
            )


def _physical_dir(
    value: str | Path,
    *,
    label: str,
    beneath: Path | None = None,
) -> Path:
    logical = Path(value).expanduser()
    _assert_no_symlink_components(logical, label=label)
    try:
        resolved = logical.resolve(strict=True)
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"{label} is unavailable") from exc
    if resolved != logical or not resolved.is_dir():
        raise FailedSelectionAdoptionError(f"{label} is not one physical directory")
    if beneath is not None:
        try:
            resolved.relative_to(beneath)
        except ValueError as exc:
            raise FailedSelectionAdoptionError(
                f"{label} escaped its authority root"
            ) from exc
    return resolved


def _physical_file(
    value: str | Path,
    *,
    label: str,
    beneath: Path | None = None,
) -> Path:
    logical = Path(value).expanduser()
    _assert_no_symlink_components(logical, label=label)
    try:
        resolved = logical.resolve(strict=True)
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"{label} is unavailable") from exc
    if resolved != logical:
        raise FailedSelectionAdoptionError(f"{label} is not one physical path")
    if beneath is not None:
        try:
            resolved.relative_to(beneath)
        except ValueError as exc:
            raise FailedSelectionAdoptionError(
                f"{label} escaped its authority root"
            ) from exc
    before = resolved.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise FailedSelectionAdoptionError(f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"cannot open {label} without following") from exc
    try:
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise FailedSelectionAdoptionError(f"{label} changed while opening")
    return resolved


def _read_fd_all(descriptor: int, *, block_size: int = 8 * 1024 * 1024) -> bytes:
    parts: list[bytes] = []
    while True:
        block = os.read(descriptor, block_size)
        if not block:
            return b"".join(parts)
        parts.append(block)


class _ArtifactTracker:
    """Hash each source through O_NOFOLLOW and retain pre/post stat identity."""

    def __init__(self, *, held_directories: _HeldDirectorySet | None = None) -> None:
        self._records: dict[Path, dict[str, Any]] = {}
        self._payloads: dict[Path, bytes] = {}
        self._held_directories = held_directories

    def add(
        self,
        path: Path,
        *,
        role: str,
        expected_sha256: str | None = None,
        keep_bytes: bool = False,
    ) -> bytes | None:
        if self._held_directories is not None:
            self._held_directories.add(
                path.parent,
                label=f"authority parent for {role}",
            )
            self._held_directories.assert_all()
        physical = _physical_file(path, label=role)
        existing = self._records.get(physical)
        if existing is not None:
            if expected_sha256 is not None and existing["sha256"] != expected_sha256:
                raise FailedSelectionAdoptionError(
                    f"conflicting expected hash for {physical}"
                )
            roles = set(existing["roles"])
            roles.add(role)
            existing["roles"] = sorted(roles)
            return self._payloads.get(physical)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(physical, flags)
        try:
            opened = os.fstat(descriptor)
            before = physical.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise FailedSelectionAdoptionError(f"{role} changed before hashing")
            digest = hashlib.sha256()
            pieces: list[bytes] | None = [] if keep_bytes else None
            while True:
                block = os.read(descriptor, 8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                if pieces is not None:
                    pieces.append(block)
            observed_sha = digest.hexdigest()
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = physical.lstat()
        before_id = _stat_identity(before)
        after_id = _stat_identity(after_path)
        if (
            before_id != after_id
            or _stat_identity(after_fd) != before_id
            or before_id["nlink"] != 1
        ):
            raise FailedSelectionAdoptionError(f"{role} changed while hashing")
        if expected_sha256 is not None and observed_sha != expected_sha256:
            raise FailedSelectionAdoptionError(f"{role} SHA256 mismatch")
        self._records[physical] = {
            "path": str(physical),
            "roles": [role],
            "sha256": observed_sha,
            "pre_stat": before_id,
            "post_stat": after_id,
            "opened_with_o_nofollow": True,
        }
        if pieces is not None:
            payload = b"".join(pieces)
            self._payloads[physical] = payload
            if self._held_directories is not None:
                self._held_directories.assert_all()
            return payload
        if self._held_directories is not None:
            self._held_directories.assert_all()
        return None

    def json(
        self,
        path: Path,
        *,
        role: str,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        raw = self.add(
            path,
            role=role,
            expected_sha256=expected_sha256,
            keep_bytes=True,
        )
        assert raw is not None
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise FailedSelectionAdoptionError(f"{role} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise FailedSelectionAdoptionError(f"{role} is not a JSON object")
        return value

    def assert_current(self) -> None:
        for path, record in self._records.items():
            try:
                current = _stat_identity(path.lstat())
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    f"adoption source disappeared: {path}"
                ) from exc
            if current != record["post_stat"]:
                raise FailedSelectionAdoptionError(
                    f"adoption source stat drifted: {path}"
                )

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._records, key=str))

    def evidence(self) -> list[dict[str, Any]]:
        return [self._records[path] for path in sorted(self._records, key=str)]


def _derive_authority_paths(profile: FailedSelectionAuthority) -> _AuthorityPaths:
    profile.validate()
    control = _physical_dir(profile.control_root, label="fixed AutoDL control root")
    namespace = _physical_dir(
        control / profile.namespace,
        label="fixed AutoDL namespace",
        beneath=control,
    )
    source_manifest = _physical_file(
        namespace / "manifests" / f"{profile.controller_id}.json",
        label="fixed controller source manifest",
        beneath=namespace,
    )
    controller_root = _physical_dir(
        namespace / profile.controller_id,
        label="fixed persistent controller root",
        beneath=namespace,
    )
    task_root = controller_root / "tasks"
    return _AuthorityPaths(
        control_root=control,
        namespace_root=namespace,
        source_manifest=source_manifest,
        controller_root=controller_root,
        controller_snapshot=_physical_file(
            controller_root / "controller_manifest.json",
            label="persistent controller snapshot",
            beneath=controller_root,
        ),
        close_gate=_physical_file(
            task_root / profile.close_task_id / "gate.json",
            label="close task gate",
            beneath=controller_root,
        ),
        close_state=_physical_file(
            task_root / profile.close_task_id / "state.json",
            label="close task state",
            beneath=controller_root,
        ),
        final_gate=_physical_file(
            task_root / profile.final_task_id / "gate.json",
            label="failed final task gate",
            beneath=controller_root,
        ),
        final_state=_physical_file(
            task_root / profile.final_task_id / "state.json",
            label="failed final task state",
            beneath=controller_root,
        ),
    )


def _task_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("tasks")
    if not isinstance(rows, list):
        raise FailedSelectionAdoptionError("controller tasks are absent")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise FailedSelectionAdoptionError("controller task row is invalid")
        task_id = str(row["id"])
        if task_id in result:
            raise FailedSelectionAdoptionError(f"duplicate controller task: {task_id}")
        result[task_id] = row
    return result


def _expected_output(task: Mapping[str, Any], attempt: int) -> Path:
    template = task.get("expected_output")
    if not isinstance(template, str) or not template:
        raise FailedSelectionAdoptionError("task expected_output is absent")
    expanded = template.replace("{attempt}", str(int(attempt)))
    if "{" in expanded or "}" in expanded:
        raise FailedSelectionAdoptionError(
            "task expected_output retains an unresolved placeholder"
        )
    logical = Path(expanded)
    if not logical.is_absolute():
        raise FailedSelectionAdoptionError("task expected_output is not absolute")
    return logical


def _task_state_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project mutable controller state onto frozen scientific authority."""

    if set(payload) != TASK_STATE_TOP_KEYS:
        raise FailedSelectionAdoptionError(
            "task-state top-level schema changed: "
            + json.dumps(
                {
                    "missing": sorted(TASK_STATE_TOP_KEYS - set(payload)),
                    "extra": sorted(set(payload) - TASK_STATE_TOP_KEYS),
                },
                sort_keys=True,
            )
        )
    instances = payload.get("instances")
    if not isinstance(instances, Mapping) or set(instances) != {"main"}:
        raise FailedSelectionAdoptionError(
            "task state does not contain exactly one main instance"
        )
    instance = instances["main"]
    if not isinstance(instance, Mapping):
        raise FailedSelectionAdoptionError("task main state is not an object")
    if set(instance) != TASK_STATE_MAIN_KEYS:
        raise FailedSelectionAdoptionError(
            "task-state main schema changed: "
            + json.dumps(
                {
                    "missing": sorted(TASK_STATE_MAIN_KEYS - set(instance)),
                    "extra": sorted(set(instance) - TASK_STATE_MAIN_KEYS),
                },
                sort_keys=True,
            )
        )
    for label, value, positive in (
        ("schema_version", payload.get("schema_version"), True),
        ("attempt", instance.get("attempt"), False),
        ("child_pid", instance.get("child_pid"), True),
        ("launcher_pid", instance.get("launcher_pid"), True),
        ("worker_pid", instance.get("worker_pid"), True),
    ):
        if type(value) is not int or (positive and value <= 0) or (
            not positive and value < 0
        ):
            raise FailedSelectionAdoptionError(
                f"task-state {label} is not a strict integer"
            )
    for label in ("launcher_identity", "worker_identity"):
        identity = instance.get(label)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != TASK_STATE_IDENTITY_KEYS
        ):
            raise FailedSelectionAdoptionError(
                f"task-state {label} schema changed"
            )
        if (
            not _is_plain_positive_int(identity.get("pid"))
            or not _is_plain_positive_int(identity.get("start_ticks"))
            or not _is_sha256(identity.get("command_sha256"))
        ):
            raise FailedSelectionAdoptionError(
                f"task-state {label} is not a strict PID identity"
            )
    if not _is_utc_timestamp(payload.get("updated_at")) or not _is_utc_timestamp(
        instance.get("heartbeat_at")
    ):
        raise FailedSelectionAdoptionError(
            "task-state mutable timestamps are not nonempty UTC timestamps"
        )
    failure_reason = instance.get("failure_reason")
    if failure_reason is None:
        failure_projection = None
    elif isinstance(failure_reason, str):
        raw_reason = failure_reason.encode("utf-8")
        failure_projection = {
            "length": len(failure_reason),
            "sha256": hashlib.sha256(raw_reason).hexdigest(),
        }
    else:
        raise FailedSelectionAdoptionError("task failure_reason is not text/null")
    projected_instance = dict(instance)
    projected_instance["launcher_identity"] = dict(instance["launcher_identity"])
    projected_instance["worker_identity"] = dict(instance["worker_identity"])
    projected_instance["required_absolute_output_files"] = list(
        instance["required_absolute_output_files"]
    ) if isinstance(instance["required_absolute_output_files"], list) else instance[
        "required_absolute_output_files"
    ]
    projected_instance["heartbeat_at"] = MUTABLE_STATE_VALUE
    projected_instance["failure_reason"] = failure_projection
    return {
        "schema_version": payload["schema_version"],
        "dataset": payload["dataset"],
        "stage": payload["stage"],
        "task_id": payload["task_id"],
        "state": payload["state"],
        "reason": payload["reason"],
        "created_at": payload["created_at"],
        "updated_at": MUTABLE_STATE_VALUE,
        "instances": {"main": projected_instance},
    }


def _observe_task_state(
    path: Path,
    *,
    expected: TaskStateAuthority,
    held_directories: _HeldDirectorySet,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one atomic controller-state generation and validate its projection.

    Controller heartbeats may atomically replace ``state.json``.  Each attempt
    therefore requires the opened descriptor and named path to identify the
    same generation; a concurrent replacement is retried, while every
    accepted generation must have the exact frozen scientific projection.
    """

    expected.validate()
    held_directories.add(path.parent, label=f"task-state parent {expected.task_id}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(8):
        held_directories.assert_all()
        try:
            before_named = path.lstat()
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise FailedSelectionAdoptionError(
                f"cannot read task state: {expected.task_id}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            raw = _read_fd_all(descriptor)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            after_named = path.lstat()
        except OSError:
            continue
        if (
            stat.S_ISREG(opened.st_mode)
            and _stat_identity(opened) == _stat_identity(after_fd)
            and (opened.st_dev, opened.st_ino)
            == (before_named.st_dev, before_named.st_ino)
            == (after_named.st_dev, after_named.st_ino)
            and opened.st_nlink == 1
        ):
            break
    else:
        raise FailedSelectionAdoptionError(
            f"task state churned throughout inspection: {expected.task_id}"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise FailedSelectionAdoptionError(
            f"task state is invalid JSON: {expected.task_id}"
        ) from exc
    if not isinstance(payload, dict):
        raise FailedSelectionAdoptionError(
            f"task state is not an object: {expected.task_id}"
        )
    projection = _task_state_projection(payload)
    projection_sha256 = _stable_hash(projection)
    if projection != expected.projection() or projection_sha256 != expected.projection_sha256:
        raise FailedSelectionAdoptionError(
            f"task-state scientific projection changed: {expected.task_id}"
        )
    held_directories.assert_all()
    return payload, {
        "path": str(path),
        "observed_sha256": hashlib.sha256(raw).hexdigest(),
        "observed_stat": _stat_identity(opened),
        "projection_sha256": projection_sha256,
        "projection": projection,
    }


def _validate_task_terminal(
    *,
    task: Mapping[str, Any],
    gate: Mapping[str, Any],
    state: Mapping[str, Any],
    expected_status: str,
    expected_failure_class: str | None,
) -> dict[str, Any]:
    task_id = str(task.get("id"))
    runs = gate.get("runs")
    instances = state.get("instances")
    if (
        gate.get("schema_version") != 1
        or gate.get("task_id") != task_id
        or gate.get("status") != expected_status
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
        or set((instances or {}).keys()) != {"main"}
        or state.get("schema_version") != 1
        or state.get("task_id") != task_id
        or state.get("state") != expected_status
    ):
        raise FailedSelectionAdoptionError(
            f"{task_id} is not one exact {expected_status} main run"
        )
    run = runs[0]
    instance = instances["main"]
    attempt = run.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 0
        or run.get("instance_id") != "main"
        or run.get("state") != expected_status
        or instance.get("instance_id") != "main"
        or instance.get("state") != expected_status
        or instance.get("attempt") != attempt
        or instance.get("run_id") != run.get("run_id")
    ):
        raise FailedSelectionAdoptionError(f"{task_id} run/state closure mismatch")
    expected = _expected_output(task, attempt)
    run_output = Path(str(run.get("expected_output") or ""))
    state_output = Path(str(instance.get("expected_output") or ""))
    if run_output != expected or state_output != expected:
        raise FailedSelectionAdoptionError(f"{task_id} output/attempt closure mismatch")
    if expected_failure_class is None:
        if instance.get("failure_class") not in {None, ""}:
            raise FailedSelectionAdoptionError(f"{task_id} PASS has a failure class")
    elif instance.get("failure_class") != expected_failure_class:
        raise FailedSelectionAdoptionError(
            f"{task_id} failure class is not {expected_failure_class}"
        )
    return {
        "task_id": task_id,
        "status": expected_status,
        "instance_id": "main",
        "attempt": attempt,
        "run_id": run.get("run_id"),
        "expected_output": str(expected),
        "failure_class": instance.get("failure_class"),
        "worker_pid": instance.get("worker_pid"),
        "child_pid": instance.get("child_pid"),
        "worker_identity": instance.get("worker_identity"),
    }


def _npy_metadata(path: Path) -> tuple[np.ndarray, tuple[int, ...], np.dtype[Any]]:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise FailedSelectionAdoptionError(f"cannot mmap ndarray: {path}") from exc
    if not isinstance(value, np.memmap):
        raise FailedSelectionAdoptionError(f"ndarray is not mmap-backed: {path}")
    return value, tuple(map(int, value.shape)), value.dtype


def _validate_pair_bitmap_semantic_equivalence(
    *,
    scan_bitmap: np.ndarray,
    materialized_bitmap: np.ndarray,
    expected_rows: int,
) -> None:
    """Validate the two independently hashed close-bitmap representations."""

    if (
        tuple(map(int, scan_bitmap.shape)) != (expected_rows,)
        or scan_bitmap.dtype != np.dtype(np.uint8)
    ):
        raise FailedSelectionAdoptionError(
            "pair-semantics scan bitmap schema changed"
        )
    if (
        tuple(map(int, materialized_bitmap.shape)) != (expected_rows,)
        or materialized_bitmap.dtype != np.dtype(np.bool_)
    ):
        raise FailedSelectionAdoptionError("close bitmap schema changed")

    for start in range(0, expected_rows, PAIR_BITMAP_COMPARE_BLOCK_ROWS):
        stop = min(start + PAIR_BITMAP_COMPARE_BLOCK_ROWS, expected_rows)
        scan_chunk = np.asarray(scan_bitmap[start:stop])
        materialized_chunk = np.asarray(materialized_bitmap[start:stop])
        if bool(np.any((scan_chunk != 0) & (scan_chunk != 1))):
            raise FailedSelectionAdoptionError(
                "pair-semantics scan bitmap contains non-binary values"
            )
        if not np.array_equal(
            scan_chunk.astype(np.bool_, copy=False),
            materialized_chunk,
        ):
            raise FailedSelectionAdoptionError(
                "pair-semantics and materialized close bitmaps are not "
                "row-wise equivalent"
            )
        if not bool(np.all(materialized_chunk)):
            raise FailedSelectionAdoptionError(
                "close bitmap contradicts the all-pairs-close authority"
            )


def _require_exact_path(value: Any, expected: Path, *, label: str) -> None:
    if not isinstance(value, str) or Path(value) != expected:
        raise FailedSelectionAdoptionError(f"{label} path changed")


def _anchor_graph_summary(
    anchor_count: int,
    edges: np.ndarray,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    """Canonical components and exact degrees for the failed anchor graph.

    Component IDs are assigned by increasing minimum anchor position.  Anchor
    positions are already the sorted global-row selection order, so this is a
    deterministic canonical label independent of graph traversal ordering.
    """

    adjacency = [set() for _ in range(anchor_count)]
    canonical: list[tuple[int, int]] = []
    for raw_left, raw_right in edges.tolist():
        left, right = int(raw_left), int(raw_right)
        if not 0 <= left < right < anchor_count:
            raise FailedSelectionAdoptionError("anchor edge is not canonical")
        adjacency[left].add(right)
        adjacency[right].add(left)
        canonical.append((left, right))
    if canonical != sorted(set(canonical)):
        raise FailedSelectionAdoptionError("anchor edge array is not sorted unique")
    unseen = set(range(anchor_count))
    sizes: list[int] = []
    labels = [-1] * anchor_count
    while unseen:
        start = min(unseen)
        component_id = len(sizes)
        reached = {start}
        frontier = [start]
        unseen.remove(start)
        while frontier:
            node = frontier.pop()
            for target in adjacency[node]:
                if target not in reached:
                    reached.add(target)
                    unseen.discard(target)
                    frontier.append(target)
        for node in reached:
            labels[node] = component_id
        sizes.append(len(reached))
    degrees_including_self = tuple(len(row) + 1 for row in adjacency)
    neighborhoods_including_self = tuple(
        tuple(sorted({index, *members}))
        for index, members in enumerate(adjacency)
    )
    if min(degrees_including_self, default=0) < 3:
        raise FailedSelectionAdoptionError(
            "disconnected anchor graph lost min_samples=3 self-neighbor closure"
        )
    return (
        tuple(sizes),
        tuple(labels),
        degrees_including_self,
        neighborhoods_including_self,
    )


def _proc_generation(proc_root: Path, pid: int) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    root = proc_root / str(pid)
    try:
        raw = (root / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"cannot audit PID {pid}") from exc
    close = raw.rfind(")")
    fields = raw[close + 2 :].split()
    if close < 0 or len(fields) <= 19:
        raise FailedSelectionAdoptionError(f"PID {pid} stat is invalid")
    try:
        start_ticks = int(fields[19])
    except ValueError as exc:
        raise FailedSelectionAdoptionError(f"PID {pid} start_ticks is invalid") from exc
    try:
        cmdline = (root / "cmdline").read_bytes().replace(b"\0", b" ").strip()
    except FileNotFoundError:
        cmdline = b""
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"cannot audit PID {pid} cmdline") from exc
    return {
        "pid": pid,
        "state": fields[0],
        "start_ticks": start_ticks,
        "command_sha256": hashlib.sha256(cmdline).hexdigest(),
        "cmdline": cmdline.decode("utf-8", errors="replace"),
    }


def _validate_worker_exited(
    *,
    terminal: Mapping[str, Any],
    final_root: Path,
    proc_root: Path,
) -> dict[str, Any]:
    raw_identity = terminal.get("worker_identity")
    worker_pid = terminal.get("worker_pid")
    if (
        not isinstance(raw_identity, Mapping)
        or not isinstance(worker_pid, int)
        or isinstance(worker_pid, bool)
        or int(raw_identity.get("pid", -1)) != worker_pid
        or int(raw_identity.get("start_ticks", -1)) <= 0
        or not _is_sha256(raw_identity.get("command_sha256"))
    ):
        raise FailedSelectionAdoptionError(
            "failed task lacks a PID-generation-bound worker identity"
        )
    current = _proc_generation(proc_root, worker_pid)
    if current is None:
        worker_observation = "ORIGINAL_GENERATION_EXITED_PID_ABSENT"
    elif current["start_ticks"] != int(raw_identity["start_ticks"]):
        worker_observation = "ORIGINAL_GENERATION_EXITED_PID_REUSED"
    elif current["state"] == "Z":
        worker_observation = "ORIGINAL_GENERATION_EXITED_ZOMBIE"
    else:
        raise FailedSelectionAdoptionError(
            "failed task worker generation is still alive; no signal was sent"
        )

    child_pid = terminal.get("child_pid")
    child_observation = "NO_RECORDED_CHILD_PID"
    if isinstance(child_pid, int) and not isinstance(child_pid, bool) and child_pid > 0:
        child = _proc_generation(proc_root, child_pid)
        if child is None:
            child_observation = "RECORDED_CHILD_PID_ABSENT"
        elif child["state"] == "Z":
            child_observation = "RECORDED_CHILD_ZOMBIE"
        else:
            # The persistent state does not carry a generation identity for
            # child_pid.  A live process at that PID therefore cannot be proved
            # to be a reuse and must fail closed regardless of its command.
            raise FailedSelectionAdoptionError(
                "recorded scientific child PID is still live or unprovably reused; "
                "no signal was sent"
            )
    return {
        "expected_worker_identity": dict(raw_identity),
        "worker_observation": worker_observation,
        "recorded_child_pid": child_pid,
        "child_observation": child_observation,
        "old_science_worker_exited": True,
        "signals_sent": [],
    }


def _validate_process_exit_receipt(
    *,
    recorded: Any,
    current: Any,
    profile: FailedSelectionAuthority,
) -> None:
    """Validate all stable exit fields while allowing safe procfs churn.

    PID absence, proven worker-PID reuse, and zombie state are all terminal
    observations.  They may legitimately change between complete authority
    scans (for example, zombie -> absent or absent -> reused).  Observation
    strings therefore need not be byte-equal, but both the receipt and the
    current scan must use only the safe enum appropriate to the exact frozen
    worker identity and recorded child PID.  Every non-observation field is
    required to agree exactly with the profile and current full scan.
    """

    if (
        not isinstance(recorded, Mapping)
        or not isinstance(current, Mapping)
        or set(recorded) != set(_PROCESS_EXIT_KEYS)
        or set(current) != set(_PROCESS_EXIT_KEYS)
    ):
        raise FailedSelectionAdoptionError(
            "terminal worker-exit receipt schema changed"
        )
    expected_worker_identity = {
        "pid": profile.final_state_authority.worker_pid,
        "start_ticks": profile.final_state_authority.worker_start_ticks,
        "command_sha256": (
            profile.final_state_authority.worker_command_sha256
        ),
    }
    expected_child_pid = profile.final_state_authority.child_pid
    for label, process in (("recorded", recorded), ("current", current)):
        if process.get("expected_worker_identity") != expected_worker_identity:
            raise FailedSelectionAdoptionError(
                f"{label} terminal worker identity changed"
            )
        if (
            type(process.get("recorded_child_pid")) is not int
            or process.get("recorded_child_pid") != expected_child_pid
        ):
            raise FailedSelectionAdoptionError(
                f"{label} terminal child PID changed"
            )
        if process.get("old_science_worker_exited") is not True:
            raise FailedSelectionAdoptionError(
                f"{label} terminal worker-exit claim changed"
            )
        if process.get("signals_sent") != []:
            raise FailedSelectionAdoptionError(
                f"{label} terminal signal-free claim changed"
            )
        if process.get("worker_observation") not in (
            _SAFE_WORKER_EXIT_OBSERVATIONS
        ):
            raise FailedSelectionAdoptionError(
                f"{label} terminal worker observation is unsafe"
            )
        expected_child_observations = (
            _SAFE_RECORDED_CHILD_EXIT_OBSERVATIONS
            if expected_child_pid > 0
            else frozenset({_NO_RECORDED_CHILD_OBSERVATION})
        )
        if process.get("child_observation") not in expected_child_observations:
            raise FailedSelectionAdoptionError(
                f"{label} terminal child observation is unsafe"
            )

    for key in (
        "expected_worker_identity",
        "recorded_child_pid",
        "old_science_worker_exited",
        "signals_sent",
    ):
        if recorded.get(key) != current.get(key):
            raise FailedSelectionAdoptionError(
                f"terminal worker-exit evidence changed: {key}"
            )


def _fd_flags(path: Path) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("flags:"):
                return int(line.split(":", 1)[1].strip(), 8)
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, ValueError) as exc:
        raise FailedSelectionAdoptionError(f"cannot audit proc fd flags: {path}") from exc
    return None


def _find_writable_references(
    paths: Sequence[Path], *, proc_root: Path
) -> list[dict[str, Any]]:
    targets: dict[tuple[int, int], str] = {}
    for path in paths:
        value = path.stat()
        targets[(int(value.st_dev), int(value.st_ino))] = str(path)
    holders: list[dict[str, Any]] = []
    try:
        processes = sorted(
            (entry for entry in proc_root.iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
    except OSError as exc:
        raise FailedSelectionAdoptionError("cannot enumerate procfs") from exc
    for process in processes:
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except (PermissionError, OSError) as exc:
            raise FailedSelectionAdoptionError(
                f"cannot enumerate procfs descriptors for PID {process.name}"
            ) from exc
        for descriptor in descriptors:
            try:
                observed = descriptor.stat()
            except FileNotFoundError:
                continue
            except (PermissionError, OSError) as exc:
                raise FailedSelectionAdoptionError(
                    f"cannot stat procfs descriptor {descriptor}"
                ) from exc
            target = targets.get((int(observed.st_dev), int(observed.st_ino)))
            if target is None:
                continue
            flags = _fd_flags(process / "fdinfo" / descriptor.name)
            if flags is None:
                # A disappearing process/FD is benign only if the FD itself is
                # now absent.  An extant uninspectable matching FD fails closed.
                if descriptor.exists():
                    raise FailedSelectionAdoptionError(
                        f"matching procfs descriptor lacks flags: {descriptor}"
                    )
                continue
            if flags & os.O_ACCMODE == os.O_RDONLY:
                continue
            holders.append(
                {
                    "pid": int(process.name),
                    "fd": int(descriptor.name),
                    "flags": flags,
                    "path": target,
                }
            )
    return holders


def _failed_tree_expected_directories(expected: Mapping[str, str]) -> set[str]:
    directories: set[str] = set()
    for relative in expected:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _enumerate_tree_at(
    *, root_descriptor: int, label: str
) -> tuple[set[str], set[str]]:
    """Enumerate a held directory recursively without re-resolving its name."""

    files: set[str] = set()
    directories: set[str] = set()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def visit(descriptor: int, prefix: str) -> None:
        before = _directory_inode_identity(os.fstat(descriptor))
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as exc:
            raise FailedSelectionAdoptionError(
                f"cannot enumerate held {label}"
            ) from exc
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise FailedSelectionAdoptionError(
                    f"unsafe directory entry in {label}"
                )
            relative = f"{prefix}/{name}" if prefix else name
            try:
                observed = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    f"{label} changed during descriptor walk"
                ) from exc
            if stat.S_ISREG(observed.st_mode):
                files.add(relative)
                continue
            if not stat.S_ISDIR(observed.st_mode):
                raise FailedSelectionAdoptionError(
                    f"{label} contains a symlink/special entry: {relative}"
                )
            directories.add(relative)
            try:
                child = os.open(name, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    f"cannot hold {label} directory: {relative}"
                ) from exc
            try:
                opened = os.fstat(child)
                if _directory_inode_identity(opened) != _directory_inode_identity(
                    observed
                ):
                    raise FailedSelectionAdoptionError(
                        f"{label} directory changed while opening: {relative}"
                    )
                visit(child, relative)
                after_named = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if _directory_inode_identity(after_named) != _directory_inode_identity(
                    opened
                ):
                    raise FailedSelectionAdoptionError(
                        f"{label} directory changed after walking: {relative}"
                    )
            finally:
                os.close(child)
        if _directory_inode_identity(os.fstat(descriptor)) != before:
            raise FailedSelectionAdoptionError(
                f"held {label} directory inode changed"
            )

    visit(root_descriptor, "")
    return files, directories


def _assert_failed_tree_inventory_at(
    *,
    root_descriptor: int,
    profile: FailedSelectionAuthority,
) -> None:
    expected = dict(profile.failed_tree_files)
    observed_files, observed_dirs = _enumerate_tree_at(
        root_descriptor=root_descriptor,
        label="failed evidence tree",
    )
    expected_dirs = _failed_tree_expected_directories(expected)
    if observed_files != set(expected) or observed_dirs != expected_dirs:
        raise FailedSelectionAdoptionError(
            "failed evidence tree inventory changed: "
            + json.dumps(
                {
                    "missing": sorted(set(expected) - observed_files),
                    "extra": sorted(observed_files - set(expected)),
                    "missing_dirs": sorted(expected_dirs - observed_dirs),
                    "extra_dirs": sorted(observed_dirs - expected_dirs),
                },
                sort_keys=True,
            )
        )


def _assert_failed_tree_inventory_now(
    *,
    root: Path,
    profile: FailedSelectionAuthority,
    expected_identity: Mapping[str, int],
) -> None:
    with _HeldDirectorySet() as held:
        physical = held.add(root, label="failed final task attempt output")
        rows = held.evidence()
        if (
            len(rows) != 1
            or rows[0].get("path") != str(root)
            or rows[0].get("identity") != dict(expected_identity)
        ):
            raise FailedSelectionAdoptionError(
                "failed evidence root inode changed before terminal publication"
            )
        _assert_failed_tree_inventory_at(
            root_descriptor=held.descriptor(physical),
            profile=profile,
        )
        held.assert_all()


def _recorded_directory_identity(
    evidence: Mapping[str, Any], *, path: Path
) -> dict[str, int]:
    rows = evidence.get("source_directory_authority")
    if not isinstance(rows, list):
        raise FailedSelectionAdoptionError(
            "source directory authority is absent"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("path") == str(path)
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("identity"), Mapping):
        raise FailedSelectionAdoptionError(
            "failed evidence root identity is absent"
        )
    identity = matches[0]["identity"]
    if set(identity) != {"device", "inode", "mode"} or any(
        type(identity[key]) is not int for key in identity
    ):
        raise FailedSelectionAdoptionError(
            "failed evidence root identity schema changed"
        )
    return dict(identity)


def _validate_failed_tree_inventory(
    *,
    root: Path,
    root_descriptor: int,
    profile: FailedSelectionAuthority,
    tracker: _ArtifactTracker,
) -> list[dict[str, str]]:
    """Require the production-profile exact failed-tree file closure."""

    expected = dict(profile.failed_tree_files)
    _assert_failed_tree_inventory_at(
        root_descriptor=root_descriptor,
        profile=profile,
    )
    result: list[dict[str, str]] = []
    for relative, digest in sorted(expected.items()):
        path = root / relative
        tracker.add(
            path,
            role=f"fixed failed-tree artifact {relative}",
            expected_sha256=digest,
        )
        result.append({"relative_path": relative, "sha256": digest})
    # A path injected after the initial walk but while tracked files are being
    # hashed cannot be absorbed by TOFU; exact inventory is re-enumerated from
    # the same held root descriptor after all 14 hashes.
    _assert_failed_tree_inventory_at(
        root_descriptor=root_descriptor,
        profile=profile,
    )
    return result


def _validate_close_authority(
    *,
    close_root: Path,
    manifest: Mapping[str, Any],
    tracker: _ArtifactTracker,
    profile: FailedSelectionAuthority,
) -> dict[str, Any]:
    close_pass = _physical_file(
        close_root / "PASS",
        label="close task PASS-last marker",
        beneath=close_root,
    )
    pass_bytes = tracker.add(
        close_pass,
        role="close task PASS-last marker",
        keep_bytes=True,
    )
    if pass_bytes != b"PASS\n":
        raise FailedSelectionAdoptionError("close task PASS-last marker changed")
    identity = manifest.get("scientific_identity")
    if (
        manifest.get("schema_version") != CLOSE_PAIR_VIEW_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("run_complete") is not True
        or manifest.get("eligible_for_dbscan") is not True
        or manifest.get("blocking_reason") is not None
        or manifest.get("scientific_identity_sha256") != _stable_hash(identity)
        or not isinstance(identity, Mapping)
        or manifest.get("physical_store_is_full_cartesian") is not True
        or manifest.get("physical_store_rows") != profile.physical_rows
        or manifest.get("logical_close_rows") != profile.physical_rows
        or manifest.get("dbscan_input_count") != profile.physical_rows
        or manifest.get("all_pairs_close") is not True
        or manifest.get("view_storage") != "zero_copy_full_cartesian"
        or manifest.get("pairs_storage") != "implicit_cartesian_v1"
        or manifest.get("large_vector_copy_materialized") is not False
        or manifest.get("recourse_vectors_recomputed") is not False
        or manifest.get("filter_operator") != FILTER_OPERATOR
        or manifest.get("pair_orientation") != PAIR_ORIENTATION
        or manifest.get("pair_axis") != "col0=parent;col1=candidate"
        or manifest.get("chunk_order") != PAIR_ORDER
        or manifest.get("dbscan_input") != "theta_close_recourse_only"
        or float(manifest.get("theta", -1)) != profile.theta
        or manifest.get("approximation_used") is not False
        or manifest.get("physical_row_indices_path") is not None
        or manifest.get("physical_row_indices_sha256") is not None
        or manifest.get("pair_indices_sha256") != profile.physical_pairs_sha256
        or manifest.get("recourse_vectors_copied_byte_exact_from_physical_rows")
        is not True
        or manifest.get("recourse_vectors_zero_copy_indexed_from_physical_rows")
        is not False
    ):
        raise FailedSelectionAdoptionError("close-pair authority semantics changed")
    contract = identity.get("contract")
    if (
        not isinstance(contract, Mapping)
        or float(contract.get("theta", -1)) != profile.theta
        or int(contract.get("parent_count", -1)) != profile.parent_count
        or int(contract.get("candidate_count", -1)) != profile.candidate_count
        or contract.get("scale_contract") != SCALE_CONTRACT
        or contract.get("normalized_distance_contract")
        != NORMALIZED_DISTANCE_CONTRACT
        or not _is_sha256(contract.get("distance_checkpoint_sha256"))
        or not _is_sha256(contract.get("embedding_checkpoint_sha256"))
        or contract.get("filter_operator") != FILTER_OPERATOR
        or contract.get("pair_orientation") != PAIR_ORIENTATION
        or contract.get("chunk_order") != PAIR_ORDER
    ):
        raise FailedSelectionAdoptionError("close-pair scientific contract changed")
    vector_path = _physical_file(
        str(identity.get("physical_vectors_path") or ""),
        label="close physical vectors",
    )
    distance_path = _physical_file(
        str(identity.get("normalized_distances_path") or ""),
        label="close normalized distances",
    )
    pair_contract_path = _physical_file(
        str(identity.get("pair_semantics_contract_path") or ""),
        label="pair-semantics contract",
    )
    if (
        identity.get("physical_vectors_sha256") != profile.physical_vectors_sha256
        or manifest.get("recourse_vectors_path") != str(vector_path)
        or manifest.get("recourse_vectors_sha256") != profile.physical_vectors_sha256
        or identity.get("pair_semantics_contract_sha256") is None
        or manifest.get("pair_indices_path") is not None
    ):
        raise FailedSelectionAdoptionError("close source/vector identity changed")
    tracker.add(
        vector_path,
        role="physical recourse vectors",
        expected_sha256=profile.physical_vectors_sha256,
    )
    tracker.add(
        distance_path,
        role="normalized distance authority",
        expected_sha256=str(identity.get("normalized_distances_sha256") or ""),
    )
    pair_contract = tracker.json(
        pair_contract_path,
        role="pair-semantics contract",
        expected_sha256=str(identity["pair_semantics_contract_sha256"]),
    )
    pair_semantics_bitmap_path = _physical_file(
        str(pair_contract.get("close_bitmap") or ""),
        label="pair-semantics close bitmap",
        beneath=pair_contract_path.parent,
    )
    tracker.add(
        pair_semantics_bitmap_path,
        role="pair-semantics close bitmap",
        expected_sha256=str(pair_contract.get("close_bitmap_hash") or ""),
    )
    vector_stat = tracker._records[vector_path]["post_stat"]
    distance_stat = tracker._records[distance_path]["post_stat"]
    pair_contract_stat = tracker._records[pair_contract_path]["post_stat"]
    if (
        identity.get("schema_version") != CLOSE_PAIR_VIEW_SCHEMA
        or identity.get("physical_store_rows") != profile.physical_rows
        or identity.get("physical_vectors_stat_identity")
        != _dbscan_stat_identity(vector_stat)
        or identity.get("normalized_distances_stat_identity")
        != _dbscan_stat_identity(distance_stat)
        or identity.get("pair_semantics_contract_stat_identity")
        != _dbscan_stat_identity(pair_contract_stat)
    ):
        raise FailedSelectionAdoptionError("close source stat identity changed")
    vector, vector_shape, vector_dtype = _npy_metadata(vector_path)
    distances, distance_shape, distance_dtype = _npy_metadata(distance_path)
    if (
        vector_shape != (profile.physical_rows, profile.vector_features)
        or vector_dtype != np.dtype(np.float32)
        or distance_shape != (profile.physical_rows,)
        or distance_dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or identity.get("physical_vectors_shape") != list(vector_shape)
        or identity.get("physical_vectors_dtype") != str(vector_dtype)
        or identity.get("normalized_distances_shape") != list(distance_shape)
        or identity.get("normalized_distances_dtype") != str(distance_dtype)
    ):
        raise FailedSelectionAdoptionError("close source array schema changed")
    del vector, distances

    bitmap_path = _physical_file(
        str(manifest.get("close_bitmap_path") or ""),
        label="close bitmap",
        beneath=close_root,
    )
    tracker.add(
        bitmap_path,
        role="close bitmap",
        expected_sha256=str(manifest.get("close_bitmap_hash") or ""),
    )
    bitmap, bitmap_shape, bitmap_dtype = _npy_metadata(bitmap_path)
    if bitmap_shape != (profile.physical_rows,) or bitmap_dtype != np.dtype(np.bool_):
        raise FailedSelectionAdoptionError("close bitmap schema changed")

    certificate_path = _physical_file(
        str(manifest.get("all_pairs_close_certificate_path") or ""),
        label="all-pairs-close certificate",
    )
    if certificate_path.parent != pair_contract_path.parent:
        raise FailedSelectionAdoptionError(
            "all-pairs-close certificate escaped the pair-semantics authority"
        )
    certificate = tracker.json(
        certificate_path,
        role="all-pairs-close certificate",
        expected_sha256=str(manifest.get("all_pairs_close_certificate_sha256") or ""),
    )
    if (
        certificate.get("schema_version") != ALL_PAIRS_CLOSE_CERTIFICATE_SCHEMA
        or certificate.get("status") != "PASS"
        or certificate.get("all_pairs_close_proven") is not True
        or certificate.get("full_distance_scan_complete") is not True
        or certificate.get("official_sample_comparison_pass") is not True
        or certificate.get("normalization_audit_pass") is not True
        or int(certificate.get("physical_store_rows", -1)) != profile.physical_rows
        or int(certificate.get("count_distance_le_theta", -1))
        != profile.physical_rows
        or int(certificate.get("count_distance_gt_theta", -1)) != 0
        or certificate.get("physical_vectors_sha256")
        != profile.physical_vectors_sha256
        or certificate.get("normalized_distances_sha256")
        != identity.get("normalized_distances_sha256")
        or float(certificate.get("theta", -1)) != profile.theta
        or certificate.get("filter_operator") != FILTER_OPERATOR
        or certificate.get("pair_orientation") != PAIR_ORIENTATION
        or certificate.get("pair_order") != PAIR_ORDER
        or certificate.get("distance_checkpoint_sha256")
        != contract.get("distance_checkpoint_sha256")
        or certificate.get("embedding_checkpoint_sha256")
        != contract.get("embedding_checkpoint_sha256")
        or certificate.get("scale_contract") != SCALE_CONTRACT
        or certificate.get("normalized_distance_contract")
        != NORMALIZED_DISTANCE_CONTRACT
        or certificate.get("approximation_used") is not False
    ):
        raise FailedSelectionAdoptionError("ALL_PAIRS_CLOSE certificate changed")

    if (
        pair_contract.get("schema_version") != AIDS_PAIR_SEMANTICS_SCHEMA
        or pair_contract.get("status") != "PASS"
        or pair_contract.get("physical_store_rows") != profile.physical_rows
        or pair_contract.get("logical_close_rows") != profile.physical_rows
        or pair_contract.get("all_pairs_close") is not True
        or pair_contract.get("pair_order") != PAIR_ORDER
        or pair_contract.get("pair_orientation") != PAIR_ORIENTATION
        or pair_contract.get("pair_axis") != ["parent_index", "candidate_index"]
        or pair_contract.get("pair_axis_all_rows_checked") is not True
        or pair_contract.get("pair_axis_mismatch_count") != 0
        or pair_contract.get("normalized_distances") != str(distance_path)
        or pair_contract.get("normalized_distances_sha256")
        != identity.get("normalized_distances_sha256")
        or pair_contract.get("physical_vectors_sha256")
        != profile.physical_vectors_sha256
        or float(pair_contract.get("theta", -1)) != profile.theta
        or int(pair_contract.get("parent_count", -1)) != profile.parent_count
        or int(pair_contract.get("candidate_count", -1))
        != profile.candidate_count
        or pair_contract.get("filter_operator") != FILTER_OPERATOR
        or pair_contract.get("scale_contract") != SCALE_CONTRACT
        or pair_contract.get("normalized_distance_contract")
        != NORMALIZED_DISTANCE_CONTRACT
        or pair_contract.get("distance_checkpoint_hash")
        != contract.get("distance_checkpoint_sha256")
        or pair_contract.get("embedding_checkpoint_hash")
        != contract.get("embedding_checkpoint_sha256")
        or pair_contract.get("source_mutated") is not False
    ):
        raise FailedSelectionAdoptionError("pair-semantics authority changed")
    pair_semantics_bitmap, semantics_bitmap_shape, semantics_bitmap_dtype = (
        _npy_metadata(pair_semantics_bitmap_path)
    )
    if (
        semantics_bitmap_shape != (profile.physical_rows,)
        or semantics_bitmap_dtype != np.dtype(np.uint8)
    ):
        raise FailedSelectionAdoptionError(
            "pair-semantics scan bitmap schema changed"
        )
    _validate_pair_bitmap_semantic_equivalence(
        scan_bitmap=pair_semantics_bitmap,
        materialized_bitmap=bitmap,
        expected_rows=profile.physical_rows,
    )
    del pair_semantics_bitmap, bitmap
    pair_manifest_path = _physical_file(
        str(pair_contract.get("source_pair_store_manifest") or ""),
        label="physical pair-store manifest",
    )
    pair_manifest = tracker.json(
        pair_manifest_path,
        role="physical pair-store manifest",
        expected_sha256=str(pair_contract.get("source_pair_store_manifest_sha256") or ""),
    )
    pairs_path = _physical_file(
        str(pair_manifest.get("pairs_path") or ""),
        label="physical pair indices",
        beneath=pair_manifest_path.parent,
    )
    pair_vectors_path = _physical_file(
        str(pair_manifest.get("vectors_path") or ""),
        label="pair-store vectors",
        beneath=pair_manifest_path.parent,
    )
    if (
        pair_manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or pair_manifest.get("run_complete") is not True
        or pair_manifest.get("candidate_major_parent_minor_order") is not True
        or int(pair_manifest.get("row_count", -1)) != profile.physical_rows
        or int(pair_manifest.get("vector_dim", -1)) != profile.vector_features
        or pair_manifest.get("vectors_dtype") != "float32"
        or pair_manifest.get("pairs_sha256") != profile.physical_pairs_sha256
        or pair_manifest.get("vectors_sha256") != profile.physical_vectors_sha256
        or pair_vectors_path != vector_path
    ):
        raise FailedSelectionAdoptionError("physical pair-store authority changed")
    tracker.add(
        pairs_path,
        role="physical pair indices",
        expected_sha256=profile.physical_pairs_sha256,
    )
    pairs, pairs_shape, pairs_dtype = _npy_metadata(pairs_path)
    if pairs_shape != (profile.physical_rows, 2) or pairs_dtype != np.dtype(np.int64):
        raise FailedSelectionAdoptionError("physical pair-index schema changed")
    del pairs
    return {
        "manifest": str(close_root / "close_pair_contract.json"),
        "manifest_sha256": profile.close_manifest_sha256,
        "status": "PASS",
        "all_pairs_close": True,
        "physical_rows": profile.physical_rows,
        "logical_close_rows": profile.physical_rows,
        "pair_axis": "col0=parent;col1=candidate",
        "pair_order": PAIR_ORDER,
        "filter_operator": FILTER_OPERATOR,
        "vector_path": str(vector_path),
        "vector_sha256": profile.physical_vectors_sha256,
        "pair_path": str(pairs_path),
        "pair_sha256": profile.physical_pairs_sha256,
        "normalized_distances_sha256": identity["normalized_distances_sha256"],
        "close_bitmap_sha256": manifest["close_bitmap_hash"],
        "pair_semantics_contract": str(pair_contract_path),
        "pair_semantics_contract_sha256": identity[
            "pair_semantics_contract_sha256"
        ],
    }


def _validate_failed_selection(
    *,
    final_root: Path,
    close: Mapping[str, Any],
    tracker: _ArtifactTracker,
    profile: FailedSelectionAuthority,
) -> dict[str, Any]:
    dbscan_root = final_root / "common_recourse" / "external_memory" / "dbscan"
    expected_paths = {
        "checkpoint": dbscan_root / "checkpoint.json",
        "selection": dbscan_root / "adaptive_anchor_selection.json",
        "failure": dbscan_root / "shortcut_failure.json",
        "failure_indices": dbscan_root
        / "adaptive_first_pass_failure_indices.npy",
        "anchor_indices": dbscan_root / "shortcut_anchor_indices.npy",
        "anchor_rows": dbscan_root / "adaptive_selected_anchor_rows.npy",
        "anchor_edges": dbscan_root / "shortcut_anchor_edges.npy",
    }
    for name, path in list(expected_paths.items()):
        expected_paths[name] = _physical_file(
            path, label=f"failed DBSCAN {name}", beneath=final_root
        )
    checkpoint = tracker.json(
        expected_paths["checkpoint"],
        role="failed DBSCAN authenticated checkpoint",
        expected_sha256=profile.checkpoint_sha256,
    )
    selection = tracker.json(
        expected_paths["selection"],
        role="adaptive selection manifest",
        expected_sha256=profile.selection_manifest_sha256,
    )
    failure = tracker.json(
        expected_paths["failure"],
        role="disconnected shortcut failure",
        expected_sha256=profile.shortcut_failure_sha256,
    )
    tracker.add(
        expected_paths["failure_indices"],
        role="adaptive first-pass failure indices",
        expected_sha256=profile.failure_indices_sha256,
    )
    tracker.add(
        expected_paths["anchor_indices"],
        role="adaptive selected anchor indices",
        expected_sha256=profile.anchor_indices_sha256,
    )
    tracker.add(
        expected_paths["anchor_rows"],
        role="adaptive selected anchor rows",
        expected_sha256=profile.anchor_rows_sha256,
    )
    tracker.add(
        expected_paths["anchor_edges"],
        role="failed disconnected anchor edges",
        expected_sha256=profile.anchor_edges_sha256,
    )

    try:
        authenticated = _load_dbscan_checkpoint(expected_paths["checkpoint"])
    except Exception as exc:
        raise FailedSelectionAdoptionError("failed DBSCAN checkpoint is unauthenticated") from exc
    if authenticated != checkpoint:
        raise FailedSelectionAdoptionError("failed DBSCAN checkpoint bytes changed")
    identity = checkpoint.get("identity")
    if (
        checkpoint.get("schema_version") != DBSCAN_SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or checkpoint.get("identity_sha256") != _dbscan_stable_hash(identity)
        or checkpoint.get("phase") != "shortcut_blocked"
        or int(checkpoint.get("next_offset", -1)) != 0
        or checkpoint.get("shortcut_approximation_used") is not False
        or checkpoint.get("adaptive_selection_manifest_sha256")
        != profile.selection_manifest_sha256
        or checkpoint.get("shortcut_failure_sha256")
        != profile.shortcut_failure_sha256
    ):
        raise FailedSelectionAdoptionError("failed DBSCAN terminal checkpoint changed")
    _require_exact_path(
        checkpoint.get("adaptive_selection_manifest_path"),
        expected_paths["selection"],
        label="adaptive selection checkpoint",
    )
    _require_exact_path(
        checkpoint.get("shortcut_failure_path"),
        expected_paths["failure"],
        label="shortcut failure checkpoint",
    )
    contract = identity.get("contract")
    shortcut = identity.get("shortcut_contract")
    vector_stat = tracker._records[Path(str(close["vector_path"]))]["post_stat"]
    if (
        identity.get("schema_version") != DBSCAN_SCHEMA_VERSION
        or identity.get("vectors_path") != close["vector_path"]
        or identity.get("vectors_sha256") != close["vector_sha256"]
        or identity.get("vectors_shape")
        != [profile.physical_rows, profile.vector_features]
        or identity.get("vectors_dtype") != "float32"
        or identity.get("vectors_stat_identity")
        != _dbscan_stat_identity(vector_stat)
        or identity.get("nearest_neighbors_fit_method") != "brute"
        or identity.get("nearest_neighbors_metric") != "euclidean"
        or not isinstance(contract, Mapping)
        or float(contract.get("eps", -1)) != profile.eps
        or int(contract.get("min_samples", -1)) != profile.min_samples
        or contract.get("shortcut_mode")
        != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        or not isinstance(shortcut, Mapping)
        or shortcut.get("mode") != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        or int(shortcut.get("seed_count", -1)) != profile.seed_count
        or shortcut.get("anchor_selection_rule")
        != "sorted_unique_union_of_seed_and_all_failure_indices_v1"
    ):
        raise FailedSelectionAdoptionError("failed DBSCAN scientific identity changed")
    try:
        ledgers = _load_progress_ledgers(
            checkpoint,
            identity=identity,
            num_samples=profile.physical_rows,
        )
    except Exception as exc:
        raise FailedSelectionAdoptionError("adaptive progress-ledger closure failed") from exc
    if set(ledgers) != {
        "adaptive_seed_scan",
        "adaptive_failure_scan",
        "shortcut_anchor_scan",
    }:
        raise FailedSelectionAdoptionError("adaptive progress-ledger phase set changed")
    for phase in ("adaptive_seed_scan", "adaptive_failure_scan"):
        ledger = ledgers[phase]
        if (
            ledger.get("complete") is not True
            or int(ledger.get("committed_offset", -1)) != profile.physical_rows
        ):
            raise FailedSelectionAdoptionError(f"{phase} ledger is incomplete")
    lower_ledger = ledgers["shortcut_anchor_scan"]
    if (
        lower_ledger.get("complete") is not False
        or int(lower_ledger.get("committed_offset", -1)) != 0
        or lower_ledger.get("entries") != []
    ):
        raise FailedSelectionAdoptionError(
            "failed anchor-scan ledger no longer stops before the disconnected graph"
        )
    try:
        anchors, validated_selection = _validate_adaptive_selection_manifest(
            path=expected_paths["selection"],
            expected_sha256=profile.selection_manifest_sha256,
            root=expected_paths["selection"].parent,
            identity=identity,
            progress_ledgers=ledgers,
        )
    except Exception as exc:
        raise FailedSelectionAdoptionError(
            "adaptive selection/seed/failure ledger closure failed"
        ) from exc
    if validated_selection != selection:
        raise FailedSelectionAdoptionError("adaptive selection bytes changed")
    selection_identity = selection.get("selection_identity")
    if (
        selection.get("schema_version") != ADAPTIVE_SELECTION_SCHEMA_VERSION
        or selection.get("run_complete") is not True
        or not isinstance(selection_identity, Mapping)
        or int(selection_identity.get("failure_count", -1)) != profile.failure_count
        or int(selection_identity.get("anchor_count", -1)) != profile.anchor_count
        or selection_identity.get("failure_indices_sha256")
        != profile.failure_indices_sha256
        or selection_identity.get("anchor_indices_sha256")
        != profile.anchor_indices_sha256
        or selection_identity.get("anchor_rows_sha256") != profile.anchor_rows_sha256
    ):
        raise FailedSelectionAdoptionError("adaptive selection identity changed")
    for field, path in (
        ("failure_indices_path", expected_paths["failure_indices"]),
        ("anchor_indices_path", expected_paths["anchor_indices"]),
        ("anchor_rows_path", expected_paths["anchor_rows"]),
    ):
        _require_exact_path(selection_identity.get(field), path, label=field)
    if len(anchors) != profile.anchor_count:
        raise FailedSelectionAdoptionError("adaptive selected anchor count changed")

    failure_indices = np.load(
        expected_paths["failure_indices"], mmap_mode="r", allow_pickle=False
    )
    anchor_indices = np.load(
        expected_paths["anchor_indices"], mmap_mode="r", allow_pickle=False
    )
    anchor_rows = np.load(expected_paths["anchor_rows"], mmap_mode="r", allow_pickle=False)
    anchor_edges = np.load(
        expected_paths["anchor_edges"], mmap_mode="r", allow_pickle=False
    )
    seed_indices = [int(value) for value in selection_identity.get("seed_indices") or []]
    if (
        failure_indices.dtype != np.dtype(np.intp)
        or failure_indices.shape != (profile.failure_count,)
        or anchor_indices.dtype != np.dtype(np.intp)
        or anchor_indices.shape != (profile.anchor_count,)
        or anchor_rows.dtype != np.dtype(np.float32)
        or anchor_rows.shape != (profile.anchor_count, profile.vector_features)
        or anchor_edges.dtype != np.dtype(np.intp)
        or anchor_edges.shape != (profile.anchor_edge_count, 2)
        or len(seed_indices) != profile.seed_count
        or not np.array_equal(failure_indices, np.unique(failure_indices))
        or not np.array_equal(anchor_indices, np.unique(anchor_indices))
        or not np.array_equal(
            anchor_indices,
            np.asarray(
                sorted(set(seed_indices).union(failure_indices.tolist())),
                dtype=np.intp,
            ),
        )
        or _sample_indices_sha256(failure_indices)
        != selection_identity.get("failure_index_list_sha256")
        or _sample_indices_sha256(anchor_indices)
        != selection_identity.get("selected_anchor_indices_sha256")
    ):
        raise FailedSelectionAdoptionError("adaptive selection arrays changed")
    (
        observed_components,
        canonical_component_labels,
        degrees_including_self,
        neighborhoods_including_self,
    ) = _anchor_graph_summary(profile.anchor_count, anchor_edges)
    if observed_components != profile.initial_component_sizes:
        raise FailedSelectionAdoptionError(
            "disconnected anchor component partition changed"
        )
    position_by_global_row = {
        int(global_row): position
        for position, global_row in enumerate(anchor_indices.tolist())
    }
    try:
        seed_component_ids = tuple(
            canonical_component_labels[position_by_global_row[seed]]
            for seed in seed_indices
        )
    except KeyError as exc:
        raise FailedSelectionAdoptionError(
            "adaptive seed is absent from the selected anchor closure"
        ) from exc
    unique_seed_component = len(set(seed_component_ids)) == 1
    if (
        not unique_seed_component
        or len(seed_component_ids) != profile.seed_count
        or observed_components[seed_component_ids[0]] != profile.seed_count
    ):
        raise FailedSelectionAdoptionError(
            "the three seeds are not the canonical size-3 anchor component"
        )
    if (
        failure.get("schema_version")
        != "comrecgc_dbscan_anchor_proof_failure_v1"
        or failure.get("status") != "INCONCLUSIVE"
        or failure.get("scientific_identity_sha256")
        != checkpoint.get("identity_sha256")
        or failure.get("reason") != "anchor_epsilon_graph_disconnected"
        or int(failure.get("num_samples", -1)) != profile.physical_rows
        or failure.get("fallback_allowed") is not False
        or failure.get("approximation_used") is not False
        or not isinstance(failure.get("details"), Mapping)
        or int(failure["details"].get("anchor_count", -1)) != profile.anchor_count
        or int(failure["details"].get("anchor_component_reached_count", -1))
        != profile.initial_component_sizes[0]
        or int(failure["details"].get("anchor_edge_count", -1))
        != profile.anchor_edge_count
        or failure["details"].get("anchor_neighborhoods_sha256")
        != hashlib.sha256(
            json.dumps(
                [list(row) for row in neighborhoods_including_self],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise FailedSelectionAdoptionError("disconnected shortcut failure changed")
    del failure_indices, anchor_indices, anchor_rows, anchor_edges
    return {
        "checkpoint": str(expected_paths["checkpoint"]),
        "checkpoint_sha256": profile.checkpoint_sha256,
        "checkpoint_phase": "shortcut_blocked",
        "selection_manifest": str(expected_paths["selection"]),
        "selection_manifest_sha256": profile.selection_manifest_sha256,
        "failure_artifact": str(expected_paths["failure"]),
        "failure_artifact_sha256": profile.shortcut_failure_sha256,
        "failure_reason": "anchor_epsilon_graph_disconnected",
        "seed_count": profile.seed_count,
        "failure_count": profile.failure_count,
        "anchor_count": profile.anchor_count,
        "anchor_edge_count": profile.anchor_edge_count,
        "initial_component_sizes": list(profile.initial_component_sizes),
        "initial_component_canonicalization": (
            "component_id_by_minimum_selected_anchor_position_v1"
        ),
        "initial_component_canonical_labels": list(canonical_component_labels),
        "initial_component_canonical_labels_sha256": _stable_hash(
            list(canonical_component_labels)
        ),
        "initial_component_canonical_labels_hash_contract": (
            "sha256_canonical_json_int_list_v1"
        ),
        "seed_component_ids": list(seed_component_ids),
        "unique_seed_component": unique_seed_component,
        "seed_component_id": seed_component_ids[0],
        "seed_component_size": observed_components[seed_component_ids[0]],
        "anchor_degrees_including_self_sha256": _stable_hash(
            list(degrees_including_self)
        ),
        "anchor_degrees_including_self_hash_contract": (
            "sha256_canonical_json_int_list_v1"
        ),
        "anchor_degree_including_self_min": min(degrees_including_self),
        "anchor_neighborhoods_sha256": failure["details"][
            "anchor_neighborhoods_sha256"
        ],
        "seed_progress_ledger_complete": True,
        "failure_progress_ledger_complete": True,
        "selection_rule": "sorted_unique_union_of_seed_and_all_failure_indices_v1",
        "approximation_used": False,
        "dbscan_partition_proven": False,
    }


def _inspect_authority(
    *, profile: FailedSelectionAuthority, proc_root: Path
) -> dict[str, Any]:
    profile.validate()
    control_path = Path(profile.control_root)
    namespace_path = control_path / profile.namespace
    controller_path = namespace_path / profile.controller_id
    task_path = controller_path / "tasks"
    with _HeldDirectorySet() as held_directories:
        # Freeze the fixed authority hierarchy before deriving or reading any
        # child file.  A rename plus byte-identical copy during path discovery
        # then fails the named-inode assertions below.
        held_directories.add(control_path, label="fixed AutoDL control root")
        held_directories.add(namespace_path, label="fixed AutoDL namespace")
        held_directories.add(
            namespace_path / "manifests",
            label="fixed AutoDL manifest namespace",
        )
        held_directories.add(controller_path, label="fixed persistent controller root")
        held_directories.add(task_path, label="fixed persistent task root")
        held_directories.add(
            task_path / profile.close_task_id,
            label="fixed close task root",
        )
        held_directories.add(
            task_path / profile.final_task_id,
            label="fixed final task root",
        )
        held_directories.assert_all()
        paths = _derive_authority_paths(profile)
        held_directories.assert_all()
        proc = _physical_dir(proc_root, label="procfs root")
        tracker = _ArtifactTracker(held_directories=held_directories)
        source_manifest = tracker.json(
            paths.source_manifest,
            role="fixed controller source manifest",
            expected_sha256=profile.controller_manifest_sha256,
        )
        snapshot = tracker.json(
            paths.controller_snapshot,
            role="persistent controller manifest snapshot",
        )
        if (
            source_manifest.get("schema_version") != 1
            or source_manifest.get("controller_id") != profile.controller_id
            or source_manifest.get("paper_frozen") is not True
            or "run_tastemolnet" in source_manifest
        ):
            raise FailedSelectionAdoptionError("source controller manifest changed")
        expected_snapshot = dict(source_manifest)
        expected_snapshot["source_manifest"] = str(paths.source_manifest)
        expected_snapshot["source_manifest_sha256"] = profile.controller_manifest_sha256
        if snapshot != expected_snapshot:
            raise FailedSelectionAdoptionError(
                "persistent controller snapshot is not the exact fixed manifest"
            )
        tasks = _task_map(source_manifest)
        if profile.close_task_id not in tasks or profile.final_task_id not in tasks:
            raise FailedSelectionAdoptionError("required close/final tasks are absent")

        close_gate = tracker.json(
            paths.close_gate,
            role="close task PASS gate",
            expected_sha256=profile.close_gate_sha256,
        )
        final_gate = tracker.json(
            paths.final_gate,
            role="failed final task gate",
            expected_sha256=profile.final_gate_sha256,
        )
        close_state_before, close_observation_before = _observe_task_state(
            paths.close_state,
            expected=profile.close_state_authority,
            held_directories=held_directories,
        )
        final_state_before, final_observation_before = _observe_task_state(
            paths.final_state,
            expected=profile.final_state_authority,
            held_directories=held_directories,
        )
        close_terminal = _validate_task_terminal(
            task=tasks[profile.close_task_id],
            gate=close_gate,
            state=close_state_before,
            expected_status="PASS",
            expected_failure_class=None,
        )
        final_terminal = _validate_task_terminal(
            task=tasks[profile.final_task_id],
            gate=final_gate,
            state=final_state_before,
            expected_status="FAILED",
            expected_failure_class="SEMANTIC",
        )
        close_root = held_directories.add(
            Path(close_terminal["expected_output"]),
            label="close task attempt output",
        )
        final_root = held_directories.add(
            Path(final_terminal["expected_output"]),
            label="failed final task attempt output",
        )
        if (
            close_root == final_root
            or close_root in final_root.parents
            or final_root in close_root.parents
        ):
            raise FailedSelectionAdoptionError("close and failed-final outputs overlap")
        held_directories.assert_all()
        close_manifest_path = _physical_file(
            close_root / "close_pair_contract.json",
            label="close task manifest",
            beneath=close_root,
        )
        close_manifest = tracker.json(
            close_manifest_path,
            role="close-pair PASS manifest",
            expected_sha256=profile.close_manifest_sha256,
        )
        close = _validate_close_authority(
            close_root=close_root,
            manifest=close_manifest,
            tracker=tracker,
            profile=profile,
        )
        failed = _validate_failed_selection(
            final_root=final_root,
            close=close,
            tracker=tracker,
            profile=profile,
        )
        if (final_root / "PASS").exists() or (
            final_root / "_RUN_COMPLETE.json"
        ).exists():
            raise FailedSelectionAdoptionError(
                "failed final source unexpectedly claims a scientific PASS"
            )
        failed_tree_inventory = _validate_failed_tree_inventory(
            root=final_root,
            root_descriptor=held_directories.descriptor(final_root),
            profile=profile,
            tracker=tracker,
        )
        process = _validate_worker_exited(
            terminal=final_terminal,
            final_root=final_root,
            proc_root=proc,
        )
        writers = _find_writable_references(tracker.paths, proc_root=proc)
        if writers:
            raise FailedSelectionAdoptionError(
                "adoption source retains writable process references: "
                + json.dumps(writers[:10], sort_keys=True)
            )
        tracker.assert_current()
        held_directories.assert_all()
        close_state_after, close_observation_after = _observe_task_state(
            paths.close_state,
            expected=profile.close_state_authority,
            held_directories=held_directories,
        )
        final_state_after, final_observation_after = _observe_task_state(
            paths.final_state,
            expected=profile.final_state_authority,
            held_directories=held_directories,
        )
        if (
            _validate_task_terminal(
                task=tasks[profile.close_task_id],
                gate=close_gate,
                state=close_state_after,
                expected_status="PASS",
                expected_failure_class=None,
            )
            != close_terminal
            or _validate_task_terminal(
                task=tasks[profile.final_task_id],
                gate=final_gate,
                state=final_state_after,
                expected_status="FAILED",
                expected_failure_class="SEMANTIC",
            )
            != final_terminal
        ):
            raise FailedSelectionAdoptionError(
                "task terminal closure drifted across authority inspection"
            )
        tracker.assert_current()
        held_directories.assert_all()
        _assert_failed_tree_inventory_at(
            root_descriptor=held_directories.descriptor(final_root),
            profile=profile,
        )
        authority = {
            "control_root": str(paths.control_root),
            "namespace": profile.namespace,
            "namespace_root": str(paths.namespace_root),
            "controller_id": profile.controller_id,
            "source_manifest": str(paths.source_manifest),
            "source_manifest_sha256": profile.controller_manifest_sha256,
            "persistent_controller_root": str(paths.controller_root),
            "persistent_controller_snapshot": str(paths.controller_snapshot),
            "close_gate": str(paths.close_gate),
            "close_gate_sha256": profile.close_gate_sha256,
            "final_gate": str(paths.final_gate),
            "final_gate_sha256": profile.final_gate_sha256,
            "authority_paths_exactly_derived": True,
            "copied_namespace_accepted": False,
        }
        return {
            "authority": authority,
            "task_state_authority": {
                "close": profile.close_state_authority.projection(),
                "close_projection_sha256": (
                    profile.close_state_authority.projection_sha256
                ),
                "final": profile.final_state_authority.projection(),
                "final_projection_sha256": (
                    profile.final_state_authority.projection_sha256
                ),
            },
            "task_state_observations": {
                "close": [close_observation_before, close_observation_after],
                "final": [final_observation_before, final_observation_after],
            },
            "close_task": close_terminal,
            "final_task": final_terminal,
            "close_authority": close,
            "failed_selection": failed,
            "failed_tree_inventory": failed_tree_inventory,
            "process_exit": process,
            "source_artifacts": tracker.evidence(),
            "source_directory_authority": held_directories.evidence(),
            "source_writable_reference_count": 0,
        }


class _OutputParentHandle:
    """Hold the fixed output parent before any lock/output name is touched."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor = -1
        self.identity: dict[str, int] | None = None

    def __enter__(self) -> "_OutputParentHandle":
        _assert_no_symlink_components(self.path, label="fixed adoption output parent")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            self.descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise FailedSelectionAdoptionError(
                "cannot hold fixed adoption output parent"
            ) from exc
        try:
            opened = os.fstat(self.descriptor)
            named = self.path.lstat()
            resolved = self.path.resolve(strict=True)
            self.identity = _directory_inode_identity(opened)
            if (
                resolved != self.path
                or not stat.S_ISDIR(opened.st_mode)
                or self.identity != _directory_inode_identity(named)
            ):
                raise FailedSelectionAdoptionError(
                    "fixed adoption output parent inode changed while opening"
                )
            self.assert_inode()
            return self
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = -1
            raise

    def assert_inode(self, *, named: bool = True) -> None:
        assert self.identity is not None and self.descriptor >= 0
        if _directory_inode_identity(os.fstat(self.descriptor)) != self.identity:
            raise FailedSelectionAdoptionError(
                "held fixed adoption output parent inode changed"
            )
        if named:
            try:
                current = self.path.lstat()
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    "named fixed adoption output parent disappeared"
                ) from exc
            if (
                not stat.S_ISDIR(current.st_mode)
                or _directory_inode_identity(current) != self.identity
            ):
                raise FailedSelectionAdoptionError(
                    "named fixed adoption output parent inode changed"
                )

    def child_exists(self, name: str) -> bool:
        self.assert_inode()
        try:
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def __exit__(self, *_args: Any) -> None:
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except OSError:
                pass
            self.descriptor = -1


class _OutputLock:
    def __init__(self, output: Path, *, parent: _OutputParentHandle) -> None:
        self.output = output
        self.parent = parent
        self.path = output.parent / f".{output.name}.failed-selection-adoption.lock"
        self.name = self.path.name
        self.descriptor = -1
        self.identity: dict[str, int] | None = None

    def __enter__(self) -> "_OutputLock":
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.parent.assert_inode()
        created = False
        try:
            self.descriptor = os.open(
                self.name,
                flags,
                dir_fd=self.parent.descriptor,
            )
        except FileNotFoundError:
            try:
                self.descriptor = os.open(
                    self.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=self.parent.descriptor,
                )
                created = True
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    "cannot create physical adoption lock in held parent"
                ) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise FailedSelectionAdoptionError(
                    "adoption lock is a symlink"
                ) from exc
            raise FailedSelectionAdoptionError(
                "cannot acquire physical adoption lock in held parent"
            ) from exc
        if created:
            os.fsync(self.parent.descriptor)
            self.parent.assert_inode()
        opened = os.fstat(self.descriptor)
        path_stat = os.stat(
            self.name,
            dir_fd=self.parent.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            os.close(self.descriptor)
            self.descriptor = -1
            raise FailedSelectionAdoptionError("adoption lock inode changed")
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.descriptor)
            self.descriptor = -1
            raise FailedSelectionAdoptionError("another adoption owner holds the lock") from exc
        self.identity = _stat_identity(opened)
        return self

    def assert_inode(self) -> None:
        assert self.identity is not None and self.descriptor >= 0
        self.parent.assert_inode()
        if (
            _stat_identity(os.fstat(self.descriptor)) != self.identity
            or _stat_identity(
                os.stat(
                    self.name,
                    dir_fd=self.parent.descriptor,
                    follow_symlinks=False,
                )
            )
            != self.identity
        ):
            raise FailedSelectionAdoptionError("held adoption lock inode changed")

    def assert_fd(self) -> None:
        assert self.identity is not None and self.descriptor >= 0
        current = os.fstat(self.descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or (int(current.st_dev), int(current.st_ino), int(current.st_mode))
            != (
                self.identity["device"],
                self.identity["inode"],
                self.identity["mode"],
            )
        ):
            raise FailedSelectionAdoptionError("held adoption lock descriptor changed")

    def __exit__(self, *_args: Any) -> None:
        if self.descriptor >= 0:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self.descriptor)
            except OSError:
                pass
            self.descriptor = -1


class _OutputRootHandle:
    """Hold the direct-child output inode from creation through publication."""

    def __init__(
        self,
        output: Path,
        *,
        parent: _OutputParentHandle,
        allow_create: bool,
    ) -> None:
        self.path = output
        self.parent_handle = parent
        self.parent = parent.path
        self.allow_create = allow_create
        self.parent_descriptor = parent.descriptor
        self.descriptor = -1
        self.parent_identity: dict[str, int] | None = parent.identity
        self.identity: dict[str, int] | None = None
        self.fresh = False

    def __enter__(self) -> "_OutputRootHandle":
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self.parent_handle.assert_inode()
        try:
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if not self.allow_create:
                raise FailedSelectionAdoptionError("terminal adoption output is absent")
            try:
                os.mkdir(self.path.name, 0o700, dir_fd=self.parent_descriptor)
            except FileExistsError as exc:
                raise FailedSelectionAdoptionError(
                    "fresh adoption output collided"
                ) from exc
            os.fsync(self.parent_descriptor)
            self.fresh = True
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        if not stat.S_ISDIR(named.st_mode):
            raise FailedSelectionAdoptionError("adoption output is not a directory")
        self.descriptor = os.open(
            self.path.name,
            directory_flags,
            dir_fd=self.parent_descriptor,
        )
        opened = os.fstat(self.descriptor)
        self.identity = _directory_inode_identity(opened)
        if self.identity != _directory_inode_identity(named):
            raise FailedSelectionAdoptionError("adoption output changed while opening")
        self.assert_inode()
        return self

    def assert_inode(self, *, named: bool = True) -> None:
        assert self.identity is not None and self.descriptor >= 0
        assert self.parent_identity is not None and self.parent_descriptor >= 0
        if (
            _directory_inode_identity(os.fstat(self.descriptor)) != self.identity
            or _directory_inode_identity(os.fstat(self.parent_descriptor))
            != self.parent_identity
        ):
            raise FailedSelectionAdoptionError("held output directory inode changed")
        if named:
            self.parent_handle.assert_inode()
            try:
                output_named = os.stat(
                    self.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise FailedSelectionAdoptionError(
                    "named adoption output disappeared"
                ) from exc
            if _directory_inode_identity(output_named) != self.identity:
                raise FailedSelectionAdoptionError("named adoption output inode changed")

    def fsync(self) -> None:
        os.fsync(self.descriptor)

    def names(self) -> set[str]:
        return set(os.listdir(self.descriptor))

    def stat_file(self, name: str) -> dict[str, int]:
        value = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise FailedSelectionAdoptionError(f"output {name} is not regular")
        return _stat_identity(value)

    def __exit__(self, *_args: Any) -> None:
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except OSError:
                pass
            self.descriptor = -1


def _write_all(descriptor: int, payload: bytes) -> None:
    cursor = 0
    while cursor < len(payload):
        cursor += os.write(descriptor, payload[cursor:])


def _atomic_noclobber_at(
    output: _OutputRootHandle,
    name: str,
    payload: bytes,
    *,
    mode: int,
) -> dict[str, int]:
    output.assert_inode()
    temporary = f".{name}.{os.getpid()}.partial"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=output.descriptor)
    except FileExistsError as exc:
        raise FailedSelectionAdoptionError(
            f"stale output partial exists: {temporary}"
        ) from exc
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=output.descriptor,
            dst_dir_fd=output.descriptor,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise FailedSelectionAdoptionError(
            f"output no-clobber collision: {name}"
        ) from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=output.descriptor)
        except FileNotFoundError:
            pass
    output.fsync()
    output.assert_inode()
    identity = output.stat_file(name)
    if identity["nlink"] != 1:
        raise FailedSelectionAdoptionError(f"output {name} has unsafe link count")
    return identity


def _read_file_at(
    output: _OutputRootHandle,
    name: str,
    *,
    expected_nlink: int = 1,
) -> tuple[bytes, dict[str, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=output.descriptor)
    except OSError as exc:
        raise FailedSelectionAdoptionError(f"output {name} is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        raw = _read_fd_all(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = output.stat_file(name)
    if (
        not stat.S_ISREG(opened.st_mode)
        or _stat_identity(opened) != _stat_identity(after)
        or _stat_identity(opened) != named
        or opened.st_nlink != expected_nlink
    ):
        raise FailedSelectionAdoptionError(f"output {name} changed while reading")
    return raw, named


def _unlink_exact_file_at(
    output: _OutputRootHandle,
    name: str,
    identity: Mapping[str, int],
) -> None:
    output.assert_inode()
    if output.stat_file(name) != dict(identity):
        raise FailedSelectionAdoptionError(f"output {name} changed before removal")
    os.unlink(name, dir_fd=output.descriptor)
    output.fsync()
    output.assert_inode()


def _unlink_if_same_inode_at(
    output: _OutputRootHandle,
    name: str,
    identity: Mapping[str, int],
) -> None:
    """Revoke the exact marker in the held original directory only."""

    output.assert_inode(named=False)
    try:
        current = output.stat_file(name)
    except FileNotFoundError:
        return
    if (current["device"], current["inode"]) != (
        int(identity["device"]),
        int(identity["inode"]),
    ):
        raise FailedSelectionAdoptionError(
            "cannot safely revoke a replaced recovery-evidence marker"
        )
    os.unlink(name, dir_fd=output.descriptor)
    output.fsync()


def _validate_output_location(
    output_dir: Path, *, profile: FailedSelectionAuthority
) -> tuple[Path, Path]:
    """Validate the lexical direct-child contract without touching child names."""

    if not output_dir.is_absolute():
        raise FailedSelectionAdoptionError("adoption output must be absolute")
    allowed = Path(profile.output_parent)
    if not allowed.is_absolute():
        raise FailedSelectionAdoptionError("fixed adoption output parent is not absolute")
    if output_dir.parent != allowed or output_dir.name in {"", ".", ".."}:
        raise FailedSelectionAdoptionError(
            "adoption output must be a direct fresh child of the fixed parent"
        )
    return allowed, allowed


def _discover_source_locations(profile: FailedSelectionAuthority) -> tuple[Path, ...]:
    """Resolve every source root/file needed for the pre-write overlap gate."""

    paths = _derive_authority_paths(profile)
    locations: set[Path] = {
        paths.control_root,
        paths.namespace_root,
        paths.controller_root,
        paths.source_manifest,
        paths.controller_snapshot,
        paths.close_gate,
        paths.close_state,
        paths.final_gate,
        paths.final_state,
    }

    def add_file(
        path: Path,
        *,
        label: str,
        beneath: Path | None = None,
    ) -> Path:
        physical = _physical_file(path, label=label, beneath=beneath)
        locations.add(physical)
        locations.add(_physical_dir(physical.parent, label=f"parent of {label}"))
        return physical

    close_root = _physical_dir(
        profile.close_state_authority.expected_output,
        label="frozen close output root",
    )
    final_root = _physical_dir(
        profile.final_state_authority.expected_output,
        label="frozen failed-final output root",
    )
    locations.update({close_root, final_root})
    tracker = _ArtifactTracker()
    close_manifest_path = add_file(
        close_root / "close_pair_contract.json",
        label="overlap-gate close manifest",
    )
    close_manifest = tracker.json(
        close_manifest_path,
        role="overlap-gate close manifest",
        expected_sha256=profile.close_manifest_sha256,
    )
    identity = close_manifest.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise FailedSelectionAdoptionError("overlap-gate close identity is absent")
    for label, value in (
        ("physical vectors", identity.get("physical_vectors_path")),
        ("normalized distances", identity.get("normalized_distances_path")),
        ("close bitmap", close_manifest.get("close_bitmap_path")),
        ("all-pairs-close certificate", close_manifest.get("all_pairs_close_certificate_path")),
    ):
        add_file(Path(str(value or "")), label=f"overlap-gate {label}")
    pair_contract_path = add_file(
        Path(str(identity.get("pair_semantics_contract_path") or "")),
        label="overlap-gate pair-semantics contract",
    )
    pair_contract = tracker.json(
        pair_contract_path,
        role="overlap-gate pair-semantics contract",
        expected_sha256=str(identity.get("pair_semantics_contract_sha256") or ""),
    )
    add_file(
        Path(str(pair_contract.get("close_bitmap") or "")),
        label="overlap-gate pair-semantics bitmap",
        beneath=pair_contract_path.parent,
    )
    pair_manifest_path = add_file(
        Path(str(pair_contract.get("source_pair_store_manifest") or "")),
        label="overlap-gate pair-store manifest",
    )
    pair_manifest = tracker.json(
        pair_manifest_path,
        role="overlap-gate pair-store manifest",
        expected_sha256=str(pair_contract.get("source_pair_store_manifest_sha256") or ""),
    )
    add_file(
        Path(str(pair_manifest.get("pairs_path") or "")),
        label="overlap-gate physical pairs",
    )
    add_file(
        Path(str(pair_manifest.get("vectors_path") or "")),
        label="overlap-gate pair-store vectors",
    )
    for relative, _digest in profile.failed_tree_files:
        add_file(final_root / relative, label=f"overlap-gate failed tree {relative}")
    return tuple(sorted(locations, key=str))


def _assert_output_disjoint_from_sources(
    output_dir: Path,
    *,
    source_locations: Sequence[Path],
) -> None:
    lock_path = output_dir.parent / f".{output_dir.name}.failed-selection-adoption.lock"
    for candidate in (output_dir, lock_path):
        for source in source_locations:
            if (
                candidate == source
                or candidate in source.parents
                or source in candidate.parents
            ):
                raise FailedSelectionAdoptionError(
                    "adoption output/lock overlaps source authority: "
                    f"{candidate} vs {source}"
                )


def _receipt_core(
    *,
    evidence: Mapping[str, Any],
    profile: FailedSelectionAuthority,
    lock: _OutputLock,
    output: _OutputRootHandle,
) -> dict[str, Any]:
    assert lock.identity is not None and output.identity is not None
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "status": "RECOVERY_ONLY_READY",
        "artifact_kind": "aids_c766_failed_selection_recovery_evidence_v3",
        "terminal_marker": READY_NAME,
        "prepared_terminal_marker": READY_PREPARED_NAME,
        "terminal_marker_link_count": 2,
        "terminal_marker_contract": "ready_v3_binds_final_receipt_sha256",
        "generic_pass_marker_created": False,
        "source_final_status": "FAILED",
        "source_final_instance": "main",
        "source_final_failure_class": "SEMANTIC",
        "failed_evidence_adopted_for_recovery_only": True,
        "ordinary_pass_dependency_eligible": False,
        "scientific_result_pass": False,
        "dbscan_partition_pass": False,
        "read_only_adoption": True,
        "source_recomputed": False,
        "source_copied": False,
        "large_payload_copied": False,
        "signals_sent": [],
        "authority_profile_sha256": _stable_hash(_profile_payload(profile)),
        "authority_profile": _profile_payload(profile),
        "lock": {
            "path": str(lock.path),
            "identity": lock.identity,
            "opened_with_o_nofollow": True,
            "exclusive_flock_held_during_publication": True,
        },
        "output": {
            "path": str(output.path),
            "parent": str(output.parent),
            "identity": output.identity,
            "parent_identity": output.parent_identity,
            "direct_child_of_fixed_parent": True,
            "source_authority_disjoint": True,
            "opened_with_o_nofollow": True,
            "held_from_creation_through_terminal_publish": True,
        },
        **dict(evidence),
    }


_RECEIPT_TOP_KEYS = frozenset(
    {
        "artifact_kind",
        "authority",
        "authority_profile",
        "authority_profile_sha256",
        "close_authority",
        "close_task",
        "created_at",
        "dbscan_partition_pass",
        "failed_evidence_adopted_for_recovery_only",
        "failed_selection",
        "failed_tree_inventory",
        "final_task",
        "generic_pass_marker_created",
        "large_payload_copied",
        "lock",
        "ordinary_pass_dependency_eligible",
        "output",
        "prepared_terminal_marker",
        "process_exit",
        "read_only_adoption",
        "schema_version",
        "scientific_result_pass",
        "signals_sent",
        "source_artifacts",
        "source_copied",
        "source_directory_authority",
        "source_final_failure_class",
        "source_final_instance",
        "source_final_status",
        "source_recomputed",
        "source_writable_reference_count",
        "status",
        "task_state_authority",
        "task_state_observations",
        "terminal_marker",
        "terminal_marker_contract",
        "terminal_marker_link_count",
    }
)


def _validate_receipt_semantics(
    receipt: Mapping[str, Any], *, terminal: bool
) -> None:
    expected_keys = set(_RECEIPT_TOP_KEYS)
    if terminal:
        expected_keys.add("terminal_reopen_task_state_observations")
    if set(receipt) != expected_keys:
        raise FailedSelectionAdoptionError(
            "failed-selection receipt top-level schema changed: "
            + json.dumps(
                {
                    "missing": sorted(expected_keys - set(receipt)),
                    "extra": sorted(set(receipt) - expected_keys),
                },
                sort_keys=True,
            )
        )
    if (
        receipt.get("schema_version") != ADOPTION_SCHEMA_VERSION
        or receipt.get("status") != "RECOVERY_ONLY_READY"
        or receipt.get("artifact_kind")
        != "aids_c766_failed_selection_recovery_evidence_v3"
        or receipt.get("terminal_marker") != READY_NAME
        or receipt.get("prepared_terminal_marker") != READY_PREPARED_NAME
        or type(receipt.get("terminal_marker_link_count")) is not int
        or receipt.get("terminal_marker_link_count") != 2
        or receipt.get("terminal_marker_contract")
        != "ready_v3_binds_final_receipt_sha256"
        or receipt.get("generic_pass_marker_created") is not False
        or receipt.get("source_final_status") != "FAILED"
        or receipt.get("source_final_instance") != "main"
        or receipt.get("source_final_failure_class") != "SEMANTIC"
        or receipt.get("failed_evidence_adopted_for_recovery_only") is not True
        or receipt.get("ordinary_pass_dependency_eligible") is not False
        or receipt.get("scientific_result_pass") is not False
        or receipt.get("dbscan_partition_pass") is not False
        or receipt.get("read_only_adoption") is not True
        or receipt.get("source_recomputed") is not False
        or receipt.get("source_copied") is not False
        or receipt.get("large_payload_copied") is not False
        or receipt.get("signals_sent") != []
        or not _is_utc_timestamp(receipt.get("created_at"))
    ):
        raise FailedSelectionAdoptionError(
            "failed-selection receipt could be mistaken for a scientific PASS"
        )


def _assert_exact_json_shape(value: Any, template: Any, *, path: str) -> None:
    """Reject every extra/missing nested key and container-shape change."""

    if isinstance(template, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(template):
            actual = set(value) if isinstance(value, Mapping) else set()
            raise FailedSelectionAdoptionError(
                f"receipt nested schema changed at {path}: "
                + json.dumps(
                    {
                        "missing": sorted(set(template) - actual),
                        "extra": sorted(actual - set(template)),
                    },
                    sort_keys=True,
                )
            )
        for key in template:
            _assert_exact_json_shape(
                value[key], template[key], path=f"{path}.{key}"
            )
        return
    if isinstance(template, list):
        if not isinstance(value, list) or len(value) != len(template):
            raise FailedSelectionAdoptionError(
                f"receipt list schema changed at {path}"
            )
        for index, expected in enumerate(template):
            _assert_exact_json_shape(
                value[index], expected, path=f"{path}[{index}]"
            )
        return
    if template is None:
        if value is not None:
            raise FailedSelectionAdoptionError(
                f"receipt scalar type changed at {path}"
            )
        return
    if type(value) is not type(template):
        raise FailedSelectionAdoptionError(
            f"receipt scalar type changed at {path}"
        )


def _validate_receipt_exact_shape(
    receipt: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    profile: FailedSelectionAuthority,
    lock: _OutputLock,
    output: _OutputRootHandle,
    terminal: bool,
) -> None:
    template = {
        **_receipt_core(
            evidence=evidence,
            profile=profile,
            lock=lock,
            output=output,
        ),
        "created_at": "2000-01-01T00:00:00+00:00",
    }
    if terminal:
        template["terminal_reopen_task_state_observations"] = evidence[
            "task_state_observations"
        ]
    _assert_exact_json_shape(receipt, template, path="receipt")


def _validate_recorded_state_observations(
    value: Any,
    *,
    profile: FailedSelectionAuthority,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"close", "final"}:
        raise FailedSelectionAdoptionError(f"{label} state observations are absent")
    task_root = (
        Path(profile.control_root)
        / profile.namespace
        / profile.controller_id
        / "tasks"
    )
    for name, expected, path in (
        (
            "close",
            profile.close_state_authority,
            task_root / profile.close_task_id / "state.json",
        ),
        (
            "final",
            profile.final_state_authority,
            task_root / profile.final_task_id / "state.json",
        ),
    ):
        rows = value.get(name)
        if not isinstance(rows, list) or len(rows) != 2:
            raise FailedSelectionAdoptionError(
                f"{label} {name} state observation count changed"
            )
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or row.get("path") != str(path)
                or not _is_sha256(row.get("observed_sha256"))
                or row.get("projection_sha256") != expected.projection_sha256
                or row.get("projection") != expected.projection()
                or not isinstance(row.get("observed_stat"), Mapping)
            ):
                raise FailedSelectionAdoptionError(
                    f"{label} {name} state observation changed"
                )


def _terminal_marker_identity_if_receipt_bound(
    *,
    output: _OutputRootHandle,
    held_lock: _OutputLock,
    profile: FailedSelectionAuthority,
) -> dict[str, int] | None:
    """Authorize revocation only for the exact receipt-bound output/lock."""

    try:
        raw, _receipt_identity = _read_file_at(output, RECEIPT_NAME)
        receipt = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(receipt, Mapping):
        return None
    try:
        _validate_receipt_semantics(receipt, terminal=True)
    except FailedSelectionAdoptionError:
        return None
    if (
        receipt.get("authority_profile") != _profile_payload(profile)
        or receipt.get("authority_profile_sha256")
        != _stable_hash(_profile_payload(profile))
    ):
        return None
    expected_output = {
        "path": str(output.path),
        "parent": str(output.parent),
        "identity": output.identity,
        "parent_identity": output.parent_identity,
        "direct_child_of_fixed_parent": True,
        "source_authority_disjoint": True,
        "opened_with_o_nofollow": True,
        "held_from_creation_through_terminal_publish": True,
    }
    recorded_lock = receipt.get("lock")
    recorded_lock_identity = (
        recorded_lock.get("identity") if isinstance(recorded_lock, Mapping) else None
    )
    current_lock_identity = held_lock.identity
    lock_inode_matches = (
        isinstance(recorded_lock, Mapping)
        and isinstance(recorded_lock_identity, Mapping)
        and isinstance(current_lock_identity, Mapping)
        and recorded_lock.get("path") == str(held_lock.path)
        and recorded_lock.get("opened_with_o_nofollow") is True
        and recorded_lock.get("exclusive_flock_held_during_publication") is True
        and all(
            recorded_lock_identity.get(key) == current_lock_identity.get(key)
            for key in ("device", "inode", "mode")
        )
    )
    if receipt.get("output") != expected_output or not lock_inode_matches:
        return None
    expected_ready = _ready_marker_bytes(raw)
    try:
        prepared_raw, prepared_identity = _read_file_at(
            output,
            READY_PREPARED_NAME,
            expected_nlink=2,
        )
        marker_raw, marker_identity = _read_file_at(
            output,
            READY_NAME,
            expected_nlink=2,
        )
    except Exception:
        return None
    if (
        prepared_raw != expected_ready
        or marker_raw != expected_ready
        or (prepared_identity["device"], prepared_identity["inode"])
        != (marker_identity["device"], marker_identity["inode"])
    ):
        return None
    return marker_identity


def _validate_output_files(
    output: _OutputRootHandle,
    *,
    receipt_name: str,
    require_ready: bool,
) -> None:
    names = output.names()
    expected = (
        {receipt_name, READY_PREPARED_NAME, READY_NAME}
        if require_ready
        else {receipt_name}
    )
    if names != expected:
        raise FailedSelectionAdoptionError(
            f"adoption output is partial or contains unexpected files: {sorted(names)}"
        )
    for name in expected:
        output.stat_file(name)


def _validate_with_profile(
    *,
    output: _OutputRootHandle,
    profile: FailedSelectionAuthority,
    proc_root: Path,
    receipt_name: str,
    require_ready: bool,
    held_lock: _OutputLock,
    observation_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output.assert_inode()
    held_lock.assert_inode()
    _validate_output_files(
        output,
        receipt_name=receipt_name,
        require_ready=require_ready,
    )
    receipt_raw, receipt_identity = _read_file_at(output, receipt_name)
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except Exception as exc:
        raise FailedSelectionAdoptionError("adoption receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise FailedSelectionAdoptionError("adoption receipt is not an object")
    terminal_receipt = receipt_name == RECEIPT_NAME or require_ready
    _validate_receipt_semantics(receipt, terminal=terminal_receipt)
    # Build the nested schema only from the fixed profile plus a fresh source
    # reopen, never from receipt keys.  Do this before any nested receipt value
    # comparisons so a re-signed extra key is classified as schema forgery.
    observed = _inspect_authority(profile=profile, proc_root=proc_root)
    _validate_receipt_exact_shape(
        receipt,
        evidence=observed,
        profile=profile,
        lock=held_lock,
        output=output,
        terminal=terminal_receipt,
    )
    _validate_recorded_state_observations(
        receipt.get("task_state_observations"),
        profile=profile,
        label="initial",
    )
    terminal_observations = receipt.get("terminal_reopen_task_state_observations")
    if receipt_name == RECEIPT_NAME or require_ready:
        _validate_recorded_state_observations(
            terminal_observations,
            profile=profile,
            label="terminal-reopen",
        )
    elif terminal_observations is not None:
        raise FailedSelectionAdoptionError(
            "preterminal receipt unexpectedly claims terminal observations"
        )
    if (
        receipt.get("authority_profile") != _profile_payload(profile)
        or receipt.get("authority_profile_sha256")
        != _stable_hash(_profile_payload(profile))
    ):
        raise FailedSelectionAdoptionError("adoption authority profile changed")
    held_lock.assert_inode()
    if receipt.get("lock") != {
        "path": str(held_lock.path),
        "identity": held_lock.identity,
        "opened_with_o_nofollow": True,
        "exclusive_flock_held_during_publication": True,
    }:
        raise FailedSelectionAdoptionError("adoption lock identity changed")
    if receipt.get("output") != {
        "path": str(output.path),
        "parent": str(output.parent),
        "identity": output.identity,
        "parent_identity": output.parent_identity,
        "direct_child_of_fixed_parent": True,
        "source_authority_disjoint": True,
        "opened_with_o_nofollow": True,
        "held_from_creation_through_terminal_publish": True,
    }:
        raise FailedSelectionAdoptionError("adoption output inode changed")
    ready_identity: dict[str, int] | None = None
    prepared_identity: dict[str, int] | None = None
    if require_ready:
        expected_ready = _ready_marker_bytes(receipt_raw)
        prepared_raw, prepared_identity = _read_file_at(
            output,
            READY_PREPARED_NAME,
            expected_nlink=2,
        )
        ready_raw, ready_identity = _read_file_at(
            output,
            READY_NAME,
            expected_nlink=2,
        )
        if (
            prepared_raw != expected_ready
            or ready_raw != expected_ready
            or (prepared_identity["device"], prepared_identity["inode"])
            != (ready_identity["device"], ready_identity["inode"])
        ):
            raise FailedSelectionAdoptionError(
                "recovery-evidence terminal marker changed"
            )
    output.assert_inode()
    held_lock.assert_inode()
    stable_keys = (
        "authority",
        "task_state_authority",
        "close_task",
        "final_task",
        "close_authority",
        "failed_selection",
        "failed_tree_inventory",
        "source_artifacts",
        "source_directory_authority",
        "source_writable_reference_count",
    )
    for key in stable_keys:
        if receipt.get(key) != observed.get(key):
            raise FailedSelectionAdoptionError(f"terminal adoption evidence changed: {key}")
    recorded_process = receipt.get("process_exit")
    current_process = observed.get("process_exit")
    _validate_process_exit_receipt(
        recorded=recorded_process,
        current=current_process,
        profile=profile,
    )
    current_receipt_raw, current_receipt_identity = _read_file_at(output, receipt_name)
    if current_receipt_raw != receipt_raw or current_receipt_identity != receipt_identity:
        raise FailedSelectionAdoptionError("adoption receipt changed during reopen")
    if ready_identity is not None and output.stat_file(READY_NAME) != ready_identity:
        raise FailedSelectionAdoptionError(
            "recovery-evidence marker changed during reopen"
        )
    if (
        prepared_identity is not None
        and output.stat_file(READY_PREPARED_NAME) != prepared_identity
    ):
        raise FailedSelectionAdoptionError(
            "prepared recovery-evidence marker changed during reopen"
        )
    output.assert_inode()
    held_lock.assert_inode()
    if observation_sink is not None:
        observation_sink.clear()
        observation_sink.update(observed["task_state_observations"])
    return dict(receipt)


def _create_or_validate_with_profile(
    *, output_dir: Path, profile: FailedSelectionAuthority, proc_root: Path
) -> dict[str, Any]:
    profile.validate()
    _output_parent, parent = _validate_output_location(output_dir, profile=profile)
    # The fixed parent is the first physical object opened by this call.  All
    # sibling lock/output operations below use its descriptor, so a named
    # rename/recreate cannot redirect publication into an attacker directory.
    with _OutputParentHandle(parent) as held_parent:
        preexisting_output = held_parent.child_exists(output_dir.name)
        if not preexisting_output:
            source_locations = _discover_source_locations(profile)
            held_parent.assert_inode()
            _assert_output_disjoint_from_sources(
                output_dir,
                source_locations=source_locations,
            )
        with _OutputLock(output_dir, parent=held_parent) as lock:
            with _OutputRootHandle(
                output_dir,
                parent=held_parent,
                allow_create=True,
            ) as output:
                output.assert_inode()
                lock.assert_inode()
                if not output.fresh:
                    marker_identity = _terminal_marker_identity_if_receipt_bound(
                        output=output,
                        held_lock=lock,
                        profile=profile,
                    )
                    try:
                        source_locations = _discover_source_locations(profile)
                        _assert_output_disjoint_from_sources(
                            output_dir,
                            source_locations=source_locations,
                        )
                        return _validate_with_profile(
                            output=output,
                            profile=profile,
                            proc_root=proc_root,
                            receipt_name=RECEIPT_NAME,
                            require_ready=True,
                            held_lock=lock,
                        )
                    except BaseException:
                        if marker_identity is not None:
                            lock.assert_fd()
                            _unlink_if_same_inode_at(
                                output,
                                READY_NAME,
                                marker_identity,
                            )
                        raise
                if output.names():
                    raise FailedSelectionAdoptionError(
                        "fresh adoption output is not empty"
                    )
                evidence = _inspect_authority(profile=profile, proc_root=proc_root)
                output.assert_inode()
                lock.assert_inode()
                receipt = {
                    **_receipt_core(
                        evidence=evidence,
                        profile=profile,
                        lock=lock,
                        output=output,
                    ),
                    "created_at": _utc_now(),
                }
                encoded = (
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                preterminal_identity = _atomic_noclobber_at(
                    output,
                    PRETERMINAL_NAME,
                    encoded,
                    mode=0o444,
                )
                # This is the complete second authority scan.  No terminal
                # marker exists yet, so interruption cannot expose READY.
                terminal_observations: dict[str, Any] = {}
                reopened = _validate_with_profile(
                    output=output,
                    profile=profile,
                    proc_root=proc_root,
                    receipt_name=PRETERMINAL_NAME,
                    require_ready=False,
                    held_lock=lock,
                    observation_sink=terminal_observations,
                )
                if reopened != receipt:
                    raise FailedSelectionAdoptionError(
                        "preterminal receipt reopen changed"
                    )
                output.fsync()
                output.assert_inode()
                lock.assert_inode()
                final_receipt = {
                    **receipt,
                    "terminal_reopen_task_state_observations": terminal_observations,
                }
                final_encoded = (
                    json.dumps(final_receipt, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                final_receipt_identity = _atomic_noclobber_at(
                    output,
                    RECEIPT_NAME,
                    final_encoded,
                    mode=0o444,
                )
                _unlink_exact_file_at(
                    output,
                    PRETERMINAL_NAME,
                    preterminal_identity,
                )
                promoted_raw, observed_promoted_identity = _read_file_at(
                    output,
                    RECEIPT_NAME,
                )
                if (
                    promoted_raw != final_encoded
                    or observed_promoted_identity != final_receipt_identity
                ):
                    raise FailedSelectionAdoptionError("promoted receipt changed")
                try:
                    promoted_payload = json.loads(promoted_raw.decode("utf-8"))
                except Exception as exc:
                    raise FailedSelectionAdoptionError(
                        "final recovery-evidence receipt is invalid"
                    ) from exc
                if promoted_payload != final_receipt:
                    raise FailedSelectionAdoptionError(
                        "final recovery-evidence receipt changed"
                    )
                _validate_receipt_semantics(final_receipt, terminal=True)
                _validate_receipt_exact_shape(
                    final_receipt,
                    evidence=evidence,
                    profile=profile,
                    lock=lock,
                    output=output,
                    terminal=True,
                )
                _validate_recorded_state_observations(
                    final_receipt["terminal_reopen_task_state_observations"],
                    profile=profile,
                    label="terminal-reopen",
                )
                output.fsync()
                output.assert_inode()
                lock.assert_inode()
                prepared_marker_identity = _atomic_noclobber_at(
                    output,
                    READY_PREPARED_NAME,
                    _ready_marker_bytes(final_encoded),
                    mode=0o444,
                )
                if output.names() != {RECEIPT_NAME, READY_PREPARED_NAME}:
                    raise FailedSelectionAdoptionError(
                        "prepublication output inventory changed"
                    )
                output.fsync()
                output.assert_inode()
                lock.assert_inode()
                _assert_failed_tree_inventory_now(
                    root=Path(profile.final_state_authority.expected_output),
                    profile=profile,
                    expected_identity=_recorded_directory_identity(
                        evidence,
                        path=Path(
                            profile.final_state_authority.expected_output
                        ),
                    ),
                )
                output.assert_inode()
                lock.assert_inode()
                # The O_EXCL hard-link creation below is the final correctness
                # and publication operation.
                try:
                    os.link(
                        READY_PREPARED_NAME,
                        READY_NAME,
                        src_dir_fd=output.descriptor,
                        dst_dir_fd=output.descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise FailedSelectionAdoptionError(
                        "recovery-evidence marker no-clobber collision"
                    ) from exc
                # READY is authority only if the same lock holder can reopen
                # the entire terminal receipt and every source authority after
                # publication.  If anything drifted after the second scan,
                # revoke only the exact READY inode just recorded above.
                try:
                    os.fsync(output.descriptor)
                    ready_identity = output.stat_file(READY_NAME)
                    if (
                        ready_identity["device"],
                        ready_identity["inode"],
                    ) != (
                        prepared_marker_identity["device"],
                        prepared_marker_identity["inode"],
                    ) or ready_identity["nlink"] != 2:
                        raise FailedSelectionAdoptionError(
                            "published recovery-evidence marker inode changed"
                        )
                    reopened_terminal = _validate_with_profile(
                        output=output,
                        profile=profile,
                        proc_root=proc_root,
                        receipt_name=RECEIPT_NAME,
                        require_ready=True,
                        held_lock=lock,
                    )
                    if reopened_terminal != final_receipt:
                        raise FailedSelectionAdoptionError(
                            "post-publication terminal receipt changed"
                        )
                    return reopened_terminal
                except BaseException:
                    lock.assert_fd()
                    _unlink_if_same_inode_at(
                        output,
                        READY_NAME,
                        prepared_marker_identity,
                    )
                    raise


def create_or_validate_aids_c766_failed_selection_adoption(
    *,
    output_dir: str | Path,
    control_root: str | Path = PRODUCTION_CONTROL_ROOT,
    proc_root: str | Path = PRODUCTION_PROC_ROOT,
) -> dict[str, Any]:
    """Publish or fully reopen the one production failed-selection receipt."""

    if Path(control_root) != PRODUCTION_CONTROL_ROOT:
        raise FailedSelectionAdoptionError("production control_root is fixed")
    if Path(proc_root) != PRODUCTION_PROC_ROOT:
        raise FailedSelectionAdoptionError("production proc_root is fixed")
    return _create_or_validate_with_profile(
        output_dir=Path(output_dir),
        profile=PRODUCTION_AUTHORITY,
        proc_root=PRODUCTION_PROC_ROOT,
    )


def validate_aids_c766_failed_selection_adoption(
    *,
    output_dir: str | Path,
    control_root: str | Path = PRODUCTION_CONTROL_ROOT,
    proc_root: str | Path = PRODUCTION_PROC_ROOT,
) -> dict[str, Any]:
    """Compatibility spelling for the typed recovery-evidence verifier."""

    return verify_aids_c766_failed_selection_recovery_evidence(
        output_dir=output_dir,
        control_root=control_root,
        proc_root=proc_root,
    )


def verify_aids_c766_failed_selection_recovery_evidence(
    *,
    output_dir: str | Path,
    control_root: str | Path = PRODUCTION_CONTROL_ROOT,
    proc_root: str | Path = PRODUCTION_PROC_ROOT,
) -> dict[str, Any]:
    """Typed terminal reopen; generic PASS consumers cannot call this by accident."""

    output = Path(output_dir)
    if Path(control_root) != PRODUCTION_CONTROL_ROOT:
        raise FailedSelectionAdoptionError("production control_root is fixed")
    if Path(proc_root) != PRODUCTION_PROC_ROOT:
        raise FailedSelectionAdoptionError("production proc_root is fixed")
    _unused, parent = _validate_output_location(output, profile=PRODUCTION_AUTHORITY)
    with _OutputParentHandle(parent) as held_parent:
        with _OutputLock(output, parent=held_parent) as lock:
            with _OutputRootHandle(
                output,
                parent=held_parent,
                allow_create=False,
            ) as held_output:
                marker_identity = _terminal_marker_identity_if_receipt_bound(
                    output=held_output,
                    held_lock=lock,
                    profile=PRODUCTION_AUTHORITY,
                )
                try:
                    source_locations = _discover_source_locations(
                        PRODUCTION_AUTHORITY
                    )
                    _assert_output_disjoint_from_sources(
                        output,
                        source_locations=source_locations,
                    )
                    return _validate_with_profile(
                        output=held_output,
                        profile=PRODUCTION_AUTHORITY,
                        proc_root=PRODUCTION_PROC_ROOT,
                        receipt_name=RECEIPT_NAME,
                        require_ready=True,
                        held_lock=lock,
                    )
                except BaseException:
                    if marker_identity is not None:
                        lock.assert_fd()
                        _unlink_if_same_inode_at(
                            held_output,
                            READY_NAME,
                            marker_identity,
                        )
                    raise


__all__ = [
    "ADOPTION_SCHEMA_VERSION",
    "CONTROL_NAMESPACE",
    "CLOSE_TASK_ID",
    "FINAL_TASK_ID",
    "FailedSelectionAdoptionError",
    "PRODUCTION_AUTHORITY",
    "PRODUCTION_CONTROL_ROOT",
    "PRODUCTION_OUTPUT_PARENT",
    "PRODUCTION_PROC_ROOT",
    "READY_NAME",
    "RECEIPT_NAME",
    "SOURCE_CONTROLLER_ID",
    "TaskStateAuthority",
    "create_or_validate_aids_c766_failed_selection_adoption",
    "validate_aids_c766_failed_selection_adoption",
    "verify_aids_c766_failed_selection_recovery_evidence",
]
