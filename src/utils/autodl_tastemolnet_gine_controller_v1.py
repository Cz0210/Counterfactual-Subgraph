"""Persistent, fail-closed supervisor for the TasteMolNet three-class GINE.

The supervisor owns one fresh CID/root and one immutable science output/state
pair.  It registers each worker generation behind the shared durable exec
startup barrier, adopts a still-live generation after controller loss, and
allows at most one process-loss retry against the exact same training-state
root.  It never signals a scientific process and publishes PASS only after a
stable terminal bundle/state scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from src.oracles.gnn_oracle import verify_checkpoint_bundle
from src.train.molecular_gnn_resume import (
    FinalizationWorkspace,
    MolecularGNNResumeError,
    MolecularGNNStateReadAuthority,
    OutputParentAuthority,
    assert_no_symlink_components,
    canonical_sha256,
    paths_overlap,
    sha256_file,
)
from src.utils.autodl_exec_startup_barrier import (
    StartupBarrierRecord,
    StartupBarrierValidationError,
    arm_exec_startup_barrier,
    reconcile_interrupted_startup_barrier_publication,
    validate_reopenable_unreleased_barrier,
    validate_startup_barrier_record,
)
from src.utils.tastemolnet_research_policy import (
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


SCHEMA = "autodl_tastemolnet_gine_persistent_controller_v2"
CLAIM_SCHEMA = "autodl_tastemolnet_gine_controller_root_claim_v2"
STATE_SCHEMA = "autodl_tastemolnet_gine_controller_state_v2"
TERMINAL_SCHEMA = "autodl_tastemolnet_gine_controller_terminal_v2"
CID_PATTERN = re.compile(r"^tastemolnet_gine_v2_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
ROOT_SENTINEL = ".controller-root-identity"
ROOT_LOCK = ".controller.lock"
ROOT_CLAIM = "controller_root_claim.json"
SPEC_NAME = "controller_spec.json"
STATE_NAME = "controller_state.json"
EVENTS_NAME = "controller_events.jsonl"
TERMINAL_NAME = "controller_terminal.json"
PASS_NAME = "PASS"
RESOURCE_DEADLINE_NAME = "resource_wait_deadline.json"
RESOURCE_DEADLINE_SCHEMA = "autodl_tastemolnet_gine_resource_deadline_v1"
PUBLISHED_ADOPTION_NAME = "published_output_resume_adoption.json"
PUBLISHED_ADOPTION_SCHEMA = "autodl_tastemolnet_published_output_adoption_v1"
TRAINER_CHILD_AUTHORITY_SCHEMA = "autodl_exp_run_trainer_child_authority_v1"
TRAINER_CHILD_AUTHORITY_NAME = "trainer_child_authority.json"
TRAINER_CHILD_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "dataset",
        "stage",
        "controller_cid",
        "controller_root",
        "project_root",
        "authority_path",
        "parent_exp_run",
        "child_registered",
        "trainer_command",
        "trainer_command_sha256",
        "barrier_record",
    }
)
PROCESS_SNAPSHOT_FIELDS = frozenset(
    {
        "pid",
        "linux_start_ticks",
        "ppid",
        "argv",
        "argv_sha256",
        "cmdline_sha256",
        "cwd",
        "exe",
        "exe_identity",
    }
)
FILE_IDENTITY_FIELDS = frozenset(
    {
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_EVENTS_BYTES = 4 * 1024**2
TERMINAL_EVENT_RESERVE_BYTES = 256 * 1024
MAX_WORKER_LOG_BYTES = 32 * 1024**2
MAX_STARTUP_GENERATIONS = 64
LINUX_PROCESS_STATES = frozenset(
    {"R", "S", "D", "Z", "T", "t", "X", "x", "K", "W", "P", "I"}
)
LINUX_EXITED_PROCESS_STATES = frozenset({"Z", "X", "x"})
FROZEN_SCIENCE_ENV_KEYS = (
    "RUN_TASTEMOLNET",
    "TASTE_RESEARCH_COMPUTE_ALLOWED",
    "TASTE_PAPER_RESULTS_ALLOWED",
    "TASTE_DATA_REDISTRIBUTION_ALLOWED",
    "TASTE_UPSTREAM_LICENSE_STATUS",
    "TASTEMOLNET_POLICY_FILE",
    "TASTEMOLNET_POLICY_SHA256",
    "TASTEMOLNET_POLICY_RECEIPT",
    "TASTEMOLNET_PREPARED_ROOT",
    "TASTEMOLNET_SPLIT_ROOT",
    "TASTEMOLNET_GRAPH_CACHE_ROOT",
    "TASTEMOLNET_GNN_FULL_OUTPUT",
    "TASTEMOLNET_GNN_TRAINING_STATE_ROOT",
    "TASTEMOLNET_GINE_CONTROLLER_CID",
    "TASTEMOLNET_GINE_CONTROLLER_ROOT",
    "TASTEMOLNET_GPU_INDEX",
    "AUTODL_RUNTIME_ROOT",
    "AUTODL_ARTIFACT_ROOT",
    "AUTODL_CONTROL_ROOT",
    "AUTODL_DATA_ROOT",
    "AUTODL_PYTHON",
    "AUTODL_MAX_GPUS",
    "AUTODL_MIN_FREE_MEMORY_MB",
    "AUTODL_IDLE_UTIL_THRESHOLD",
    "AUTODL_IDLE_STABLE_SECONDS",
    "PRIMARY_GNN_BACKBONE",
    "PRIMARY_SEED",
    "MIN_PERSISTENT_FREE_GB",
    "MIN_FREE_AFTER_RESERVATIONS_GB",
    "TASTEMOLNET_STORAGE_RESERVATION_GB",
    "TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS",
    "TASTEMOLNET_GPU_WAIT_POLL_SECONDS",
    "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "NVIDIA_TF32_OVERRIDE",
    "CUDNN_DETERMINISTIC",
)
FROZEN_BASE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
)
REQUIRED_OUTPUT_FILES = (
    "model.pt",
    "last.pt",
    "last_checkpoint.json",
    "checkpoint_reload.json",
    "model_card.json",
    "feature_schema.json",
    "training_metrics.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
    "data_use_policy_binding.json",
    "graph_cache_usage.json",
    "oracle_manifest.json",
    "sha256sums.txt",
)


class TasteGINEControllerError(RuntimeError):
    """Raised when persistent controller authority or worker state drifts."""


class _ProcessGenerationExited(TasteGINEControllerError):
    """The exact Linux PID generation reached an exited procfs state."""


@dataclass(frozen=True, slots=True)
class _LinuxProcessStatObservation:
    pid: int
    state: str
    ppid: int
    start_ticks: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    _reconcile_publication_temps(path.parent, path.name)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        data = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode()
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    _reconcile_publication_temps(path.parent, path.name)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        data = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode()
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_new(path: Path, text: str) -> None:
    _reconcile_publication_temps(path.parent, path.name)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _reconcile_publication_temps(parent: Path, final_name: str) -> None:
    temporary_paths = sorted(parent.glob(f".{final_name}.*.tmp"))
    for temporary in temporary_paths:
        info = os.lstat(temporary)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink not in {1, 2}
        ):
            raise TasteGINEControllerError("unsafe interrupted publication temp")
    for temporary in temporary_paths:
        temporary.unlink()
    if temporary_paths:
        _fsync_directory(parent)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteGINEControllerError(f"{label} is not one physical file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identity = lambda value: (  # noqa: E731
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise TasteGINEControllerError(f"{label} changed while read")
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise TasteGINEControllerError(f"{label} must contain one JSON object")
    return payload


def _load_text_bound(path: Path, *, label: str) -> tuple[str, dict[str, int]]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        named_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _file_identity(before) != _file_identity(named_before)
        ):
            raise TasteGINEControllerError(f"{label} is not one physical file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(value) for value in chunks) > 64 * 1024:
                raise TasteGINEControllerError(f"{label} exceeds its read bound")
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(named_after)
        ):
            raise TasteGINEControllerError(f"{label} changed while read")
        return b"".join(chunks).decode("utf-8"), _file_identity(after)
    finally:
        os.close(descriptor)


def _directory_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
    }


def _file_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _file_binding(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "nlink": int(info.st_nlink),
    }


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _terminal_signal_present(
    root: Path, state: Mapping[str, Any] | None
) -> bool:
    """Treat PASS phase or any terminal-named artifact as terminal authority.

    A partially published terminal is never repaired by a permissive status or
    resume path.  Its very presence switches every caller to the shared strict,
    read-only validator, where a missing peer artifact fails closed.
    """

    if isinstance(state, Mapping) and state.get("phase") == "PASS":
        return True
    return any(
        path.name == PASS_NAME or "terminal" in path.name.casefold()
        for path in root.iterdir()
    )


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    result = tuple(argv)
    if not result or any(not value or "\x00" in value for value in result):
        raise TasteGINEControllerError("worker argv must be nonempty and NUL-free")
    return result


@dataclass(frozen=True, slots=True)
class TasteGINEControllerSpec:
    cid: str
    controller_root: Path
    project_root: Path
    output_dir: Path
    training_state_root: Path
    worker_argv: tuple[str, ...]
    source_identity: Mapping[str, Any]
    environment_authority: Mapping[str, str]
    config_files: tuple[Mapping[str, Any], ...]
    max_attempts: int = 2
    poll_seconds: float = 30.0
    terminal_stability_seconds: float = 2.0
    resource_wait_deadline_seconds: int = 604800

    @classmethod
    def build(
        cls,
        *,
        cid: str,
        controller_root: str | Path,
        project_root: str | Path,
        output_dir: str | Path,
        training_state_root: str | Path,
        worker_argv: Sequence[str],
        max_attempts: int = 2,
        poll_seconds: float = 30.0,
        terminal_stability_seconds: float = 2.0,
    ) -> "TasteGINEControllerSpec":
        if not CID_PATTERN.fullmatch(cid):
            raise TasteGINEControllerError("Taste controller CID is malformed")
        root = _absolute(controller_root)
        project = Path(project_root).expanduser().resolve(strict=True)
        output = _absolute(output_dir)
        state = _absolute(training_state_root)
        for label, path in (
            ("Taste controller root", root),
            ("Taste output root", output),
            ("Taste training-state root", state),
        ):
            try:
                assert_no_symlink_components(path, label=label)
            except MolecularGNNResumeError as exc:
                raise TasteGINEControllerError(str(exc)) from exc
        requested_worker = _validate_argv(worker_argv)
        bash_executable = Path("/bin/bash").resolve(strict=True)
        worker_wrapper = (
            project / "scripts/autodl/run_tastemolnet_gnn_full.sh"
        ).resolve(strict=True)
        if (
            len(requested_worker) != 2
            or requested_worker[0] not in {"bash", str(bash_executable)}
            or requested_worker[1] != str(worker_wrapper)
        ):
            raise TasteGINEControllerError("controller worker must be the reviewed Taste wrapper")
        worker = (str(bash_executable), str(worker_wrapper))
        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", "-C", str(project), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise TasteGINEControllerError("Taste controller requires a clean immutable worktree")
        source_identity = {
            "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "worker_program_path": str(bash_executable),
            "worker_program_sha256": sha256_file(bash_executable),
            "worker_wrapper_path": str(worker_wrapper),
            "worker_wrapper_sha256": sha256_file(worker_wrapper),
            "controller_module_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_executable_sha256": sha256_file(
                Path(sys.executable).resolve(strict=True)
            ),
        }
        environment_authority = {
            key: str(os.environ.get(key, ""))
            for key in (*FROZEN_SCIENCE_ENV_KEYS, *FROZEN_BASE_ENV_KEYS)
        }
        expected_flags = {
            "RUN_TASTEMOLNET": "1",
            "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
            "TASTE_PAPER_RESULTS_ALLOWED": "1",
            "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
            "TASTE_UPSTREAM_LICENSE_STATUS": "NOT_EXPLICITLY_STATED",
            "AUTODL_MAX_GPUS": "4",
            "TASTEMOLNET_GPU_INDEX": "1",
            "TASTEMOLNET_STORAGE_RESERVATION_GB": "20",
            "PRIMARY_GNN_BACKBONE": "gine",
            "PRIMARY_SEED": "7",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "7",
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUDNN_DETERMINISTIC": "1",
        }
        if any(environment_authority[key] != value for key, value in expected_flags.items()):
            raise TasteGINEControllerError("Taste scoped environment authority is incomplete")
        try:
            minimum_free_gb = int(environment_authority["MIN_PERSISTENT_FREE_GB"])
        except ValueError as exc:
            raise TasteGINEControllerError(
                "Taste persistent free-space threshold must be an integer"
            ) from exc
        if minimum_free_gb < 100:
            raise TasteGINEControllerError(
                "Taste persistent free-space threshold must be at least 100 GiB"
            )
        try:
            minimum_after_reservations_gb = int(
                environment_authority["MIN_FREE_AFTER_RESERVATIONS_GB"]
            )
        except ValueError as exc:
            raise TasteGINEControllerError(
                "Taste post-reservation free-space floor must be an integer"
            ) from exc
        if minimum_after_reservations_gb < 100:
            raise TasteGINEControllerError(
                "Taste post-reservation free-space floor must be at least 100 GiB"
            )
        expected_paths = {
            "TASTEMOLNET_GNN_FULL_OUTPUT": str(output),
            "TASTEMOLNET_GNN_TRAINING_STATE_ROOT": str(state),
            "TASTEMOLNET_GINE_CONTROLLER_CID": cid,
            "TASTEMOLNET_GINE_CONTROLLER_ROOT": str(root),
            "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT": str(
                root / PUBLISHED_ADOPTION_NAME
            ),
        }
        if any(environment_authority[key] != value for key, value in expected_paths.items()):
            raise TasteGINEControllerError("Taste controller/science environment paths drifted")
        if (
            Path(environment_authority["AUTODL_PYTHON"]).resolve(strict=True)
            != Path(sys.executable).resolve(strict=True)
        ):
            raise TasteGINEControllerError(
                "Taste worker Python must equal the controller interpreter"
            )
        policy = load_tastemolnet_research_policy(
            environment_authority["TASTEMOLNET_POLICY_FILE"],
            expected_file_sha256=environment_authority["TASTEMOLNET_POLICY_SHA256"],
        )
        policy.require_main_route()
        authority = validate_tastemolnet_local_authority(
            policy,
            prepared_root=environment_authority["TASTEMOLNET_PREPARED_ROOT"],
            graph_cache_root=environment_authority["TASTEMOLNET_GRAPH_CACHE_ROOT"],
        )
        receipt = validate_tastemolnet_policy_receipt(
            environment_authority["TASTEMOLNET_POLICY_RECEIPT"],
            policy=policy,
            authority=authority,
            require_active=True,
            require_policy_version=2,
        )
        environment_authority.update(
            {
                "policy_canonical_sha256": policy.canonical_sha256,
                "policy_receipt_sha256": receipt.sha256,
                "private_authority_sha256": stable_sha256(authority.evidence()),
            }
        )
        config_paths = (
            project / "configs/hpc.yaml",
            project / "configs/gnn/gine.yaml",
            project / "configs/autodl/tastemolnet_gine_research_v1.yaml",
        )
        config_files = tuple(
            {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
            for path in config_paths
        )
        from src.utils.env import load_and_merge_config_files

        merged_config = load_and_merge_config_files(config_paths)
        if (
            not isinstance(merged_config.get("gnn"), Mapping)
            or merged_config["gnn"].get("backbone") != "gine"
            or not isinstance(merged_config.get("training"), Mapping)
            or int(merged_config["training"].get("primary_seed", -1)) != 7
            or not isinstance(merged_config.get("autodl"), Mapping)
            or merged_config["autodl"].get("schema_version")
            != "tastemolnet_gine_research_autodl_v2"
            or merged_config["autodl"].get("backbone") != "gine"
            or merged_config["autodl"].get("classifier_family") != "gine"
            or merged_config["autodl"].get("physical_gpu_index") != 1
            or merged_config["autodl"].get("policy_file_sha256")
            != policy.file_sha256
            or merged_config["autodl"].get("prepared_output_manifest_sha256")
            != authority.prepared_output_manifest_sha256
            or merged_config["autodl"].get("split_manifest_sha256")
            != authority.split_manifest_sha256
        ):
            raise TasteGINEControllerError(
                "verified Taste configuration does not bind GINE/seed-7"
            )
        source_identity["verified_backbone_config_path"] = str(
            config_paths[1].resolve(strict=True)
        )
        source_identity["verified_backbone_config_sha256"] = sha256_file(
            config_paths[1]
        )
        if max_attempts != 2:
            raise TasteGINEControllerError("Taste controller freezes exactly two attempts")
        if poll_seconds <= 0 or terminal_stability_seconds < 0:
            raise TasteGINEControllerError("controller timing values are invalid")
        try:
            resource_wait_deadline_seconds = int(
                environment_authority["TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS"]
                or "604800"
            )
        except ValueError as exc:
            raise TasteGINEControllerError(
                "Taste resource-wait deadline must be an integer"
            ) from exc
        if resource_wait_deadline_seconds <= 0:
            raise TasteGINEControllerError(
                "Taste resource-wait deadline must be positive"
            )
        for left, right in ((root, output), (root, state), (output, state)):
            if paths_overlap(left, right):
                raise TasteGINEControllerError("controller/output/state roots must be disjoint")
        if any(paths_overlap(project, path) for path in (root, output, state)):
            raise TasteGINEControllerError(
                "controller/output/state roots must stay outside the immutable project"
            )
        return cls(
            cid=cid,
            controller_root=root,
            project_root=project,
            output_dir=output,
            training_state_root=state,
            worker_argv=worker,
            source_identity=source_identity,
            environment_authority=environment_authority,
            config_files=config_files,
            max_attempts=max_attempts,
            poll_seconds=float(poll_seconds),
            terminal_stability_seconds=float(terminal_stability_seconds),
            resource_wait_deadline_seconds=resource_wait_deadline_seconds,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "cid": self.cid,
            "controller_root": str(self.controller_root),
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "training_state_root": str(self.training_state_root),
            "worker_argv": list(self.worker_argv),
            "worker_argv_sha256": stable_sha256(list(self.worker_argv)),
            "source_identity": dict(self.source_identity),
            "environment_authority": dict(self.environment_authority),
            "environment_authority_sha256": stable_sha256(
                dict(self.environment_authority)
            ),
            "config_files": [dict(row) for row in self.config_files],
            "max_attempts": self.max_attempts,
            "poll_seconds": self.poll_seconds,
            "terminal_stability_seconds": self.terminal_stability_seconds,
            "resource_wait_deadline_seconds": self.resource_wait_deadline_seconds,
            "physical_gpu_index": 1,
            "retry_policy": "one_process_loss_retry_same_training_state_root",
            "required_output_files": list(REQUIRED_OUTPUT_FILES),
            "terminal_marker": "[TASTE_GINE_THREE_CLASS_PASS]",
            "verified_model_route": {
                "backbone": "gine",
                "seed": 7,
                "minimum_persistent_free_gb": int(
                    self.environment_authority.get("MIN_PERSISTENT_FREE_GB", "-1")
                ),
                "backbone_config_path": self.source_identity.get(
                    "verified_backbone_config_path"
                ),
                "backbone_config_sha256": self.source_identity.get(
                    "verified_backbone_config_sha256"
                ),
            },
        }


def _read_linux_process_stat(
    pid: int,
) -> _LinuxProcessStatObservation | None:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise TasteGINEControllerError("worker proc stat is unreadable") from exc
    except OSError as exc:
        raise TasteGINEControllerError(
            "worker proc stat could not be verified"
        ) from exc
    except UnicodeDecodeError as exc:
        raise TasteGINEControllerError("malformed worker proc stat") from exc
    opening = raw.find("(")
    closing = raw.rfind(")")
    fields = [] if closing < 0 else raw[closing + 2 :].split()
    try:
        observed_pid = int(raw[:opening].strip()) if opening > 0 else -1
        state = fields[0]
        ppid = int(fields[1])
        start_ticks = int(fields[19])
    except (IndexError, TypeError, ValueError) as exc:
        raise TasteGINEControllerError("malformed worker proc stat") from exc
    if (
        opening <= 0
        or closing <= opening
        or observed_pid != pid
        or len(fields) < 20
        or state not in LINUX_PROCESS_STATES
        or ppid < 0
        or start_ticks <= 0
    ):
        raise TasteGINEControllerError("malformed worker proc stat")
    return _LinuxProcessStatObservation(
        pid=observed_pid,
        state=state,
        ppid=ppid,
        start_ticks=start_ticks,
    )


def _linux_process_stat(pid: int) -> tuple[int, int] | None:
    observed = _read_linux_process_stat(pid)
    if observed is None or observed.state in LINUX_EXITED_PROCESS_STATES:
        return None
    return observed.ppid, observed.start_ticks


def _linux_start_ticks(pid: int) -> int | None:
    observed = _linux_process_stat(pid)
    return None if observed is None else observed[1]


def _exact_live_linux_generation(
    pid: int, expected_start: Any
) -> _LinuxProcessStatObservation | None:
    if not _strict_int(expected_start) or expected_start <= 0:
        raise TasteGINEControllerError(
            "worker generation Linux start ticks are untyped"
        )
    observed = _read_linux_process_stat(pid)
    if (
        observed is None
        or observed.start_ticks != expected_start
        or observed.state in LINUX_EXITED_PROCESS_STATES
    ):
        return None
    return observed


def _process_snapshot(pid: int) -> dict[str, Any]:
    if sys.platform.startswith("linux"):
        live_empty_cmdline = False
        live_incomplete_identity = False
        for _ in range(20):
            before = _read_linux_process_stat(pid)
            if (
                before is None
                or before.state in LINUX_EXITED_PROCESS_STATES
            ):
                raise _ProcessGenerationExited(
                    "worker exited before process snapshot"
                )
            ppid = before.ppid
            ticks = before.start_ticks
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            except PermissionError as exc:
                raise TasteGINEControllerError(
                    "live worker process cmdline is unreadable"
                ) from exc
            except OSError as exc:
                after_error = _read_linux_process_stat(pid)
                if (
                    after_error is None
                    or after_error.start_ticks != ticks
                    or after_error.state in LINUX_EXITED_PROCESS_STATES
                ):
                    raise _ProcessGenerationExited(
                        "worker exited while reading process cmdline"
                    ) from exc
                live_incomplete_identity = True
                time.sleep(0.001)
                continue
            if not cmdline.rstrip(b"\0"):
                after_empty = _read_linux_process_stat(pid)
                if (
                    after_empty is None
                    or after_empty.start_ticks != ticks
                    or after_empty.state in LINUX_EXITED_PROCESS_STATES
                ):
                    raise _ProcessGenerationExited(
                        "worker exited while its cmdline was empty"
                    )
                live_empty_cmdline = True
                time.sleep(0.001)
                continue
            try:
                cwd = os.path.realpath(f"/proc/{pid}/cwd")
                exe = os.path.realpath(f"/proc/{pid}/exe")
                exe_info = os.stat(f"/proc/{pid}/exe")
                cmdline_after = Path(f"/proc/{pid}/cmdline").read_bytes()
                cwd_after = os.path.realpath(f"/proc/{pid}/cwd")
                exe_after = os.path.realpath(f"/proc/{pid}/exe")
                exe_info_after = os.stat(f"/proc/{pid}/exe")
            except PermissionError as exc:
                raise TasteGINEControllerError(
                    "live worker process identity is unreadable"
                ) from exc
            except OSError as exc:
                after_error = _read_linux_process_stat(pid)
                if (
                    after_error is None
                    or after_error.start_ticks != ticks
                    or after_error.state in LINUX_EXITED_PROCESS_STATES
                ):
                    raise _ProcessGenerationExited(
                        "worker exited while reading process identity"
                    ) from exc
                live_incomplete_identity = True
                time.sleep(0.001)
                continue
            after = _read_linux_process_stat(pid)
            if (
                after is None
                or after.start_ticks != ticks
                or after.state in LINUX_EXITED_PROCESS_STATES
            ):
                raise _ProcessGenerationExited(
                    "worker exited during process snapshot"
                )
            if (
                after.ppid == ppid
                and cmdline_after == cmdline
                and cwd_after == cwd
                and exe_after == exe
                and _file_identity(exe_info_after) == _file_identity(exe_info)
            ):
                argv = [
                    value.decode("utf-8", errors="surrogateescape")
                    for value in cmdline.rstrip(b"\0").split(b"\0")
                    if value
                ]
                return {
                    "pid": pid,
                    "linux_start_ticks": ticks,
                    "ppid": ppid,
                    "argv": argv,
                    "argv_sha256": stable_sha256(argv),
                    "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
                    "cwd": cwd,
                    "exe": exe,
                    "exe_identity": _file_identity(exe_info),
                }
            time.sleep(0.001)
        if live_empty_cmdline:
            raise TasteGINEControllerError(
                "live worker process cmdline remained empty"
            )
        if live_incomplete_identity:
            raise TasteGINEControllerError(
                "live worker process identity remained unavailable"
            )
        raise TasteGINEControllerError(
            "worker process changed throughout the bounded snapshot window"
        )
    # Local non-Linux tests cannot safely adopt after controller loss.  A live
    # child is still monitored by its Popen handle in the same invocation.
    return {
        "pid": pid,
        "linux_start_ticks": None,
        "ppid": None,
        "argv": [],
        "argv_sha256": None,
        "cmdline_sha256": None,
        "cwd": str(Path.cwd()),
        "exe": None,
        "exe_identity": None,
    }


def _snapshot_live_linux_generation(
    pid: int, expected_start: Any, *, label: str
) -> dict[str, Any] | None:
    """Snapshot only a live exact generation before argv phase validation."""

    if _exact_live_linux_generation(pid, expected_start) is None:
        return None
    try:
        snapshot = _process_snapshot(pid)
    except (TasteGINEControllerError, OSError):
        if _exact_live_linux_generation(pid, expected_start) is None:
            return None
        raise
    if _exact_live_linux_generation(pid, expected_start) is None:
        return None
    if (
        snapshot.get("pid") != pid
        or snapshot.get("linux_start_ticks") != expected_start
    ):
        raise TasteGINEControllerError(
            f"live {label} PID/start snapshot changed"
        )
    argv = snapshot.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in argv
        )
    ):
        raise TasteGINEControllerError(
            f"live {label} process argv is empty or malformed"
        )
    return snapshot


def _argv_flag_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = list(argv).index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _expected_exp_run_argv(
    spec: TasteGINEControllerSpec,
    *,
    gpu_uuid: str,
    input_manifest: str | None,
    resume_training: bool,
    published_adoption: bool = False,
) -> tuple[str, ...]:
    environment = spec.environment_authority
    # argv preserves the exact frozen launcher token.  /proc/exe is validated
    # separately against its resolved physical executable in
    # ``_classify_process_phase``.  Resolving this token here would reject the
    # reviewed ``.../bin/python -> python3.10`` invocation even though its
    # executable identity is exact.
    python = environment["AUTODL_PYTHON"]
    project = str(spec.project_root.resolve(strict=True))
    exp_run = str((spec.project_root / "scripts/autodl/exp_run.py").resolve(strict=True))
    train_script = str(
        (spec.project_root / "scripts/train_molecular_gnn.py").resolve(strict=True)
    )
    backbone = environment["PRIMARY_GNN_BACKBONE"]
    hpc_config = str(spec.project_root / "configs/hpc.yaml")
    gnn_config = str(spec.project_root / f"configs/gnn/{backbone}.yaml")
    autodl_config = str(
        spec.project_root / "configs/autodl/tastemolnet_gine_research_v1.yaml"
    )
    prefix = [
        python,
        exp_run,
        "--project-root",
        project,
        "--data-root",
        environment["AUTODL_DATA_ROOT"],
        "launch",
        "--dataset",
        "tastemolnet",
        "--stage",
        "TASTEMOLNET_GINE_FULL_RESEARCH_V1",
        "--gpu-index",
        "1",
        "--gpu-uuid",
        gpu_uuid,
        "--gpu-required",
        "--heavy",
        "--max-gpus",
        "4",
        "--gpu-hard-limit",
        "4",
        "--foreground",
        "--config-file",
        hpc_config,
        "--config-file",
        gnn_config,
        "--config-file",
        autodl_config,
    ]
    if input_manifest is not None:
        prefix.extend(("--input-manifest", input_manifest))
    if published_adoption:
        prefix.extend(
            (
                "--resume-published-output-receipt",
                environment["TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT"],
            )
        )
    prefix.extend(("--expected-output", str(spec.output_dir)))
    for name in REQUIRED_OUTPUT_FILES:
        prefix.extend(("--required-output-file", name))
    prefix.extend(
        (
            "--required-log-marker",
            "[TASTE_GINE_THREE_CLASS_PASS]",
            "--",
            python,
            train_script,
            "--config",
            hpc_config,
            "--config",
            gnn_config,
            "--config",
            autodl_config,
            "--dataset",
            "tastemolnet",
            "--data-dir",
            environment["TASTEMOLNET_SPLIT_ROOT"],
            "--output-dir",
            str(spec.output_dir),
            "--profile",
            "full",
            "--device",
            "cuda:0",
            "--backbone",
            backbone,
            "--seed",
            environment["PRIMARY_SEED"],
            "--graph-cache-root",
            environment["TASTEMOLNET_GRAPH_CACHE_ROOT"],
            "--taste-policy-file",
            environment["TASTEMOLNET_POLICY_FILE"],
            "--taste-policy-sha256",
            environment["TASTEMOLNET_POLICY_SHA256"],
            "--taste-policy-receipt",
            environment["TASTEMOLNET_POLICY_RECEIPT"],
            "--taste-prepared-root",
            environment["TASTEMOLNET_PREPARED_ROOT"],
            "--training-state-dir",
            str(spec.training_state_root),
        )
    )
    if resume_training:
        prefix.append("--resume-training")
    if published_adoption:
        prefix.extend(
            (
                "--resume-published-output-receipt",
                environment["TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT"],
            )
        )
    return tuple(prefix)


def _classify_process_phase(
    snapshot: Mapping[str, Any],
    *,
    spec: TasteGINEControllerSpec,
    barrier_record: Mapping[str, Any],
) -> str:
    argv = tuple(str(value) for value in snapshot.get("argv", ()))
    cwd = str(snapshot.get("cwd"))
    exe = str(snapshot.get("exe"))
    if cwd != str(spec.project_root.resolve(strict=True)):
        raise TasteGINEControllerError("worker process cwd differs from frozen project")
    launcher = tuple(str(value) for value in barrier_record.get("launcher_argv", ()))
    python_executable = str(Path(sys.executable).resolve(strict=True))
    if argv == launcher and exe == python_executable:
        return "startup_launcher"
    target = tuple(spec.worker_argv)
    target_executable = str(Path(target[0]).resolve(strict=True))
    if argv == target and exe == target_executable:
        return "worker_target"
    expected_python_raw = spec.environment_authority.get("AUTODL_PYTHON")
    if expected_python_raw:
        expected_python = str(Path(expected_python_raw).resolve(strict=True))
        gpu_uuid = _argv_flag_value(argv, "--gpu-uuid")
        input_manifest = _argv_flag_value(argv, "--input-manifest")
        adoption_receipt = _argv_flag_value(
            argv, "--resume-published-output-receipt"
        )
        split_root = Path(spec.environment_authority["TASTEMOLNET_SPLIT_ROOT"])
        allowed_manifests = {
            None,
            str(split_root / "split_manifest.json"),
            str(split_root / "splits/split_manifest.json"),
            str(split_root / "manifest.json"),
        }
        allowed_commands = {
            _expected_exp_run_argv(
                spec,
                gpu_uuid=gpu_uuid or "",
                input_manifest=input_manifest,
                resume_training=False,
            ),
            _expected_exp_run_argv(
                spec,
                gpu_uuid=gpu_uuid or "",
                input_manifest=input_manifest,
                resume_training=True,
            ),
        }
        expected_adoption_path = spec.environment_authority.get(
            "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT"
        )
        if expected_adoption_path:
            allowed_commands.add(
                _expected_exp_run_argv(
                    spec,
                    gpu_uuid=gpu_uuid or "",
                    input_manifest=input_manifest,
                    resume_training=True,
                    published_adoption=True,
                )
            )
        if (
            exe == expected_python
            and isinstance(gpu_uuid, str)
            and re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid)
            and input_manifest in allowed_manifests
            and argv in allowed_commands
            and adoption_receipt in {None, expected_adoption_path}
        ):
            return "exp_run_target"
    raise TasteGINEControllerError("worker process argv/exe is not an allowed exec phase")


def _process_generation(
    pid: int,
    *,
    spec: TasteGINEControllerSpec | None = None,
    barrier_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _process_snapshot(pid)
    if sys.platform.startswith("linux") and spec is not None and barrier_record is not None:
        phase = _classify_process_phase(
            snapshot, spec=spec, barrier_record=barrier_record
        )
    else:
        phase = "same_invocation_uninspectable"
    return {
        "pid": pid,
        "linux_start_ticks": snapshot["linux_start_ticks"],
        "registered": snapshot,
        "registered_phase": phase,
        "last_observed": snapshot,
        "last_observed_phase": phase,
        "phase_bindings": {phase: snapshot},
        "ancestry": {
            "registered_ppid": snapshot.get("ppid"),
            "last_observed_ppid": snapshot.get("ppid"),
            "parent_pid": snapshot.get("ppid"),
            "parent_linux_start_ticks": (
                _linux_start_ticks(int(snapshot["ppid"]))
                if sys.platform.startswith("linux")
                and isinstance(snapshot.get("ppid"), int)
                else None
            ),
            "orphan_adopted": False,
        },
    }


def _observe_generation(
    generation: Mapping[str, Any],
    *,
    spec: TasteGINEControllerSpec,
    barrier_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    pid = int(generation.get("pid", -1))
    expected_start = generation.get("linux_start_ticks")
    if pid <= 0:
        return None
    if not sys.platform.startswith("linux"):
        return dict(generation) if _generation_alive(generation) else None
    if not _strict_int(expected_start) or expected_start <= 0:
        raise TasteGINEControllerError(
            "registered worker Linux start ticks are untyped"
        )
    if _linux_start_ticks(pid) != expected_start:
        return None
    registered = generation.get("registered")
    registered_phase = str(generation.get("registered_phase"))
    raw_bindings = generation.get("phase_bindings")
    ancestry_raw = generation.get("ancestry")
    if (
        not isinstance(registered, Mapping)
        or not isinstance(raw_bindings, Mapping)
        or not isinstance(ancestry_raw, Mapping)
        or registered.get("pid") != pid
        or registered.get("linux_start_ticks") != expected_start
        or registered_phase not in raw_bindings
        or raw_bindings.get(registered_phase) != registered
    ):
        raise TasteGINEControllerError(
            "registered worker PID/start/cwd/cmd/exe binding changed"
        )
    snapshot = _snapshot_live_linux_generation(
        pid, expected_start, label="worker"
    )
    if snapshot is None:
        return None
    phase = _classify_process_phase(
        snapshot, spec=spec, barrier_record=barrier_record
    )
    order = {
        "startup_launcher": 0,
        "worker_target": 1,
        "exp_run_target": 2,
    }
    previous = str(generation.get("last_observed_phase"))
    if previous not in order or order[phase] < order[previous]:
        raise TasteGINEControllerError("worker exec phase regressed")
    bindings = {
        str(key): dict(value)
        for key, value in dict(raw_bindings).items()
        if isinstance(value, Mapping)
    }
    if phase in bindings and _process_identity_without_ancestry(
        bindings[phase]
    ) != _process_identity_without_ancestry(snapshot):
        raise TasteGINEControllerError("worker process phase binding changed")
    bindings.setdefault(phase, snapshot)
    ancestry = dict(ancestry_raw)
    parent_pid = int(ancestry.get("parent_pid", -1))
    parent_start = ancestry.get("parent_linux_start_ticks")
    current_ppid = snapshot.get("ppid")
    previous_ppid = ancestry.get("last_observed_ppid")
    if current_ppid != previous_ppid:
        if (
            _linux_start_ticks(parent_pid) == parent_start
            or previous_ppid != parent_pid
        ):
            raise TasteGINEControllerError(
                "worker ancestry changed while its registered controller remained live"
            )
        ancestry["orphan_adopted"] = True
        ancestry["orphan_adopted_ppid"] = current_ppid
    ancestry["last_observed_ppid"] = current_ppid
    return {
        **dict(generation),
        "last_observed": snapshot,
        "last_observed_phase": phase,
        "phase_bindings": bindings,
        "ancestry": ancestry,
    }


def _generation_alive(generation: Mapping[str, Any]) -> bool:
    pid = int(generation.get("pid", -1))
    expected = generation.get("linux_start_ticks")
    if pid <= 0:
        return False
    if sys.platform.startswith("linux"):
        return _linux_start_ticks(pid) == expected
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_identity_without_ancestry(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "pid",
        "linux_start_ticks",
        "argv",
        "argv_sha256",
        "cmdline_sha256",
        "cwd",
        "exe",
        "exe_identity",
    )
    return {key: snapshot.get(key) for key in keys}


def _classify_trainer_phase(
    snapshot: Mapping[str, Any], *, barrier_record: Mapping[str, Any]
) -> str:
    argv = tuple(str(value) for value in snapshot.get("argv", ()))
    exe = str(snapshot.get("exe"))
    launcher = tuple(str(value) for value in barrier_record.get("launcher_argv", ()))
    target = tuple(str(value) for value in barrier_record.get("target_argv", ()))
    python = str(Path(sys.executable).resolve(strict=True))
    if argv == launcher and exe == python:
        return "trainer_startup_launcher"
    if argv == target and target:
        target_executable = str(Path(target[0]).resolve(strict=True))
        if exe == target_executable:
            return "trainer_target"
    raise TasteGINEControllerError(
        "trainer child argv/cmd/exe is not an allowed exec phase"
    )


def _validate_process_snapshot_structure(
    raw: Any, *, label: str
) -> dict[str, Any]:
    """Validate durable process evidence without consulting a live process."""

    if not isinstance(raw, Mapping) or set(raw) != PROCESS_SNAPSHOT_FIELDS:
        raise TasteGINEControllerError(f"{label} fields differ")
    snapshot = dict(raw)
    for field in ("pid", "linux_start_ticks", "ppid"):
        if not _strict_int(snapshot[field]):
            raise TasteGINEControllerError(f"{label} {field} is untyped")
    if (
        snapshot["pid"] <= 0
        or snapshot["linux_start_ticks"] <= 0
        or snapshot["ppid"] < 0
    ):
        raise TasteGINEControllerError(f"{label} process identity is invalid")
    argv = snapshot["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in argv
        )
        or snapshot["argv_sha256"] != stable_sha256(argv)
    ):
        raise TasteGINEControllerError(f"{label} argv binding is invalid")
    if not isinstance(snapshot["cmdline_sha256"], str) or not SHA256_PATTERN.fullmatch(
        snapshot["cmdline_sha256"]
    ):
        raise TasteGINEControllerError(f"{label} cmdline hash is invalid")
    for field in ("cwd", "exe"):
        value = snapshot[field]
        if (
            not isinstance(value, str)
            or not value
            or "\0" in value
            or not os.path.isabs(value)
            or os.path.normpath(value) != value
        ):
            raise TasteGINEControllerError(f"{label} {field} is invalid")
    identity = snapshot["exe_identity"]
    if not isinstance(identity, Mapping) or set(identity) != FILE_IDENTITY_FIELDS:
        raise TasteGINEControllerError(f"{label} executable identity fields differ")
    if any(not _strict_int(identity[field]) for field in FILE_IDENTITY_FIELDS):
        raise TasteGINEControllerError(f"{label} executable identity is untyped")
    if (
        identity["device"] < 0
        or identity["inode"] <= 0
        or identity["uid"] < 0
        or identity["nlink"] <= 0
        or identity["size"] <= 0
        or identity["mtime_ns"] < 0
        or identity["ctime_ns"] < 0
        or not stat.S_ISREG(identity["mode"])
    ):
        raise TasteGINEControllerError(f"{label} executable identity is invalid")
    return snapshot


def _load_trainer_child_authority_structure(
    path: Path, *, spec: TasteGINEControllerSpec
) -> tuple[dict[str, Any], dict[str, int], str]:
    """Read and strictly validate generation-independent authority fields.

    This deliberately does not bind ``parent_exp_run`` to the controller's
    current worker.  A prior, conclusively dead generation may have a different
    parent.  Every typed/hash/path relationship is nevertheless checked before
    its declared PID/start pair is allowed to classify it as historical.
    """

    try:
        assert_no_symlink_components(
            path, label="trainer child authority path"
        )
    except MolecularGNNResumeError as exc:
        raise TasteGINEControllerError(str(exc)) from exc
    try:
        text, authority_identity = _load_text_bound(
            path, label="trainer child authority"
        )
        authority = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGINEControllerError(
            "trainer child authority is not valid JSON"
        ) from exc
    if not isinstance(authority, dict) or set(authority) != (
        TRAINER_CHILD_AUTHORITY_FIELDS
    ):
        raise TasteGINEControllerError("trainer child authority fields differ")
    if (
        authority_identity["uid"] != os.getuid()
        or stat.S_IMODE(authority_identity["mode"]) != 0o600
    ):
        raise TasteGINEControllerError(
            "trainer child authority is not owner-bound mode-0600 evidence"
        )
    control_root = _absolute(spec.environment_authority["AUTODL_CONTROL_ROOT"])
    allowed_runs_root = control_root / "experiment_registry/run_state"
    try:
        relative = path.relative_to(allowed_runs_root)
    except ValueError as exc:
        raise TasteGINEControllerError(
            "trainer child authority is outside the frozen run-state root"
        ) from exc
    if len(relative.parts) != 2 or relative.name != TRAINER_CHILD_AUTHORITY_NAME:
        raise TasteGINEControllerError("trainer child authority path is not canonical")
    project_root = str(spec.project_root.resolve(strict=True))
    if (
        authority["schema_version"] != TRAINER_CHILD_AUTHORITY_SCHEMA
        or authority["status"] != "RELEASE_AUTHORIZED"
        or authority["run_id"] != relative.parts[0]
        or authority["dataset"] != "tastemolnet"
        or authority["stage"] != "TASTEMOLNET_GINE_FULL_RESEARCH_V1"
        or authority["controller_cid"] != spec.cid
        or authority["controller_root"] != str(spec.controller_root)
        or authority["project_root"] != project_root
        or authority["authority_path"] != str(path)
    ):
        raise TasteGINEControllerError("trainer child authority fields changed")

    parent = _validate_process_snapshot_structure(
        authority["parent_exp_run"], label="trainer parent snapshot"
    )
    child = _validate_process_snapshot_structure(
        authority["child_registered"], label="trainer child snapshot"
    )
    command = authority["trainer_command"]
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in command
        )
        or authority["trainer_command_sha256"] != stable_sha256(command)
    ):
        raise TasteGINEControllerError("trainer command binding is invalid")
    raw_barrier = authority["barrier_record"]
    if not isinstance(raw_barrier, Mapping):
        raise TasteGINEControllerError("trainer startup barrier is untyped")
    try:
        barrier = StartupBarrierRecord.from_mapping(raw_barrier)
    except StartupBarrierValidationError as exc:
        raise TasteGINEControllerError(
            "trainer startup barrier structure changed"
        ) from exc
    if barrier.to_dict() != dict(raw_barrier):
        raise TasteGINEControllerError("trainer startup barrier values are untyped")
    expected_python = str(Path(sys.executable).resolve(strict=True))
    if (
        barrier.schema != "autodl_exec_startup_barrier_v1"
        or barrier.kind != "durable_exec_startup_barrier"
        or barrier.state != "ARMED_UNRELEASED"
        or Path(barrier.lock_path) != path.parent / "trainer-startup.lock"
        or Path(barrier.record_path) != path.parent / "trainer-startup.json"
        or str(Path(barrier.python_executable).resolve(strict=True))
        != expected_python
        or barrier.release_token_bytes <= 0
        or not SHA256_PATTERN.fullmatch(barrier.release_token_sha256)
        or barrier.target_argv != tuple(command)
        or barrier.target_argv_sha256 != stable_sha256(command)
        or barrier.launcher_argv_sha256
        != stable_sha256(list(barrier.launcher_argv))
        or child["argv"] != list(barrier.launcher_argv)
        or child["exe"] != expected_python
        or child["cwd"] != project_root
        or child["ppid"] != parent["pid"]
        or parent["cwd"] != project_root
        or parent["exe"] != expected_python
    ):
        raise TasteGINEControllerError(
            "trainer child authority structural binding changed"
        )
    return (
        authority,
        authority_identity,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _verify_trainer_child_authority_evidence(
    path: Path, *, identity: Mapping[str, Any], sha256: str
) -> None:
    try:
        assert_no_symlink_components(
            path, label="trainer child authority evidence path"
        )
    except MolecularGNNResumeError as exc:
        raise TasteGINEControllerError(str(exc)) from exc
    text, observed_identity = _load_text_bound(
        path, label="trainer child authority evidence"
    )
    if (
        observed_identity != identity
        or hashlib.sha256(text.encode("utf-8")).hexdigest() != sha256
    ):
        raise TasteGINEControllerError(
            "trainer child authority changed during generation classification"
        )


def _declared_trainer_child_is_live(child: Mapping[str, Any]) -> bool:
    """Classify only an exact PID/start pair; ambiguity always fails closed."""

    pid = int(child["pid"])
    expected_start = int(child["linux_start_ticks"])
    if not sys.platform.startswith("linux"):
        raise TasteGINEControllerError(
            "trainer generation liveness cannot be proven without Linux /proc"
        )
    if not Path("/proc/self/stat").is_file():
        raise TasteGINEControllerError(
            "trainer generation liveness cannot be proven without mounted /proc"
        )
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError as exc:
        raise TasteGINEControllerError(
            "trainer generation proc stat is unreadable"
        ) from exc
    except OSError as exc:
        raise TasteGINEControllerError(
            "trainer generation proc stat could not be verified"
        ) from exc
    except UnicodeDecodeError as exc:
        raise TasteGINEControllerError(
            "trainer generation proc stat is malformed"
        ) from exc
    opening = raw.find("(")
    closing = raw.rfind(")")
    fields = [] if closing < 0 else raw[closing + 2 :].split()
    try:
        observed_pid = int(raw[:opening].strip()) if opening > 0 else -1
    except ValueError as exc:
        raise TasteGINEControllerError(
            "trainer generation proc stat is malformed"
        ) from exc
    if opening <= 0 or closing <= opening or observed_pid != pid or len(fields) < 20:
        raise TasteGINEControllerError("trainer generation proc stat is malformed")
    try:
        state = fields[0]
        observed_start = int(fields[19])
    except (TypeError, ValueError) as exc:
        raise TasteGINEControllerError(
            "trainer generation proc stat is malformed"
        ) from exc
    if state not in {"R", "S", "D", "Z", "T", "t", "X", "x", "K", "W", "P", "I"}:
        raise TasteGINEControllerError("trainer generation proc stat is malformed")
    if observed_start <= 0:
        raise TasteGINEControllerError("trainer generation proc stat is malformed")
    if state in {"Z", "X", "x"}:
        return False
    return observed_start == expected_start


def _trainer_generation_from_authority(
    path: Path,
    *,
    spec: TasteGINEControllerSpec,
    worker_generation: Mapping[str, Any],
    worker_barrier_record: Mapping[str, Any],
) -> dict[str, Any]:
    authority, authority_identity, authority_sha256 = (
        _load_trainer_child_authority_structure(path, spec=spec)
    )
    parent = authority.get("parent_exp_run")
    child = authority.get("child_registered")
    inner_barrier = authority.get("barrier_record")
    command = authority.get("trainer_command")
    assert isinstance(parent, Mapping)
    assert isinstance(child, Mapping)
    assert isinstance(inner_barrier, Mapping)
    assert isinstance(command, list)
    parent_pid = int(parent.get("pid", -1))
    parent_start = parent.get("linux_start_ticks")
    if (
        parent_pid != int(worker_generation.get("pid", -1))
        or parent_start != worker_generation.get("linux_start_ticks")
        or _classify_process_phase(
            parent, spec=spec, barrier_record=worker_barrier_record
        )
        != "exp_run_target"
        or int(child.get("pid", -1)) <= 0
        or child.get("linux_start_ticks") is None
        or child.get("ppid") != parent_pid
        or child.get("cwd") != str(spec.project_root.resolve(strict=True))
        or _classify_trainer_phase(child, barrier_record=inner_barrier)
        != "trainer_startup_launcher"
    ):
        raise TasteGINEControllerError(
            "trainer child PID/start/cwd/cmd/exe/ancestry binding changed"
        )
    record = validate_startup_barrier_record(
        str(inner_barrier.get("record_path")),
        expected_target_argv=[str(value) for value in command],
    )
    if record.to_dict() != dict(inner_barrier):
        raise TasteGINEControllerError("trainer startup barrier record changed")
    _verify_trainer_child_authority_evidence(
        path, identity=authority_identity, sha256=authority_sha256
    )
    phase = "trainer_startup_launcher"
    return {
        "pid": int(child["pid"]),
        "linux_start_ticks": child["linux_start_ticks"],
        "registered": dict(child),
        "registered_phase": phase,
        "last_observed": dict(child),
        "last_observed_phase": phase,
        "phase_bindings": {phase: dict(child)},
        "ancestry": {
            "registered_ppid": parent_pid,
            "last_observed_ppid": parent_pid,
            "parent_pid": parent_pid,
            "parent_linux_start_ticks": parent_start,
            "orphan_adopted": False,
        },
        "authority_path": str(path),
        "authority_sha256": authority_sha256,
        "authority_identity": authority_identity,
        "barrier_record": dict(inner_barrier),
    }


def _observe_trainer_generation(
    generation: Mapping[str, Any],
) -> dict[str, Any] | None:
    pid = int(generation.get("pid", -1))
    expected_start = generation.get("linux_start_ticks")
    if pid <= 0:
        return None
    if not sys.platform.startswith("linux"):
        return dict(generation) if _generation_alive(generation) else None
    if not _strict_int(expected_start) or expected_start <= 0:
        raise TasteGINEControllerError(
            "registered trainer Linux start ticks are untyped"
        )
    if _linux_start_ticks(pid) != expected_start:
        return None
    authority_path = Path(str(generation.get("authority_path", "")))
    if (
        _file_identity(os.lstat(authority_path))
        != generation.get("authority_identity")
        or sha256_file(authority_path) != generation.get("authority_sha256")
    ):
        raise TasteGINEControllerError("trainer child authority hash changed")
    barrier_record = generation.get("barrier_record")
    registered = generation.get("registered")
    bindings_raw = generation.get("phase_bindings")
    ancestry_raw = generation.get("ancestry")
    if not all(
        isinstance(value, Mapping)
        for value in (barrier_record, registered, bindings_raw, ancestry_raw)
    ):
        raise TasteGINEControllerError("trainer child generation is untyped")
    snapshot = _snapshot_live_linux_generation(
        pid, expected_start, label="trainer child"
    )
    if snapshot is None:
        return None
    phase = _classify_trainer_phase(snapshot, barrier_record=barrier_record)
    order = {"trainer_startup_launcher": 0, "trainer_target": 1}
    previous_phase = str(generation.get("last_observed_phase"))
    if previous_phase not in order or order[phase] < order[previous_phase]:
        raise TasteGINEControllerError("trainer child exec phase regressed")
    bindings = {
        str(key): dict(value)
        for key, value in bindings_raw.items()
        if isinstance(value, Mapping)
    }
    if phase in bindings and _process_identity_without_ancestry(
        bindings[phase]
    ) != _process_identity_without_ancestry(snapshot):
        raise TasteGINEControllerError("trainer child phase identity changed")
    bindings.setdefault(phase, dict(snapshot))
    ancestry = dict(ancestry_raw)
    parent_pid = int(ancestry.get("parent_pid", -1))
    parent_start = ancestry.get("parent_linux_start_ticks")
    parent_alive = _linux_start_ticks(parent_pid) == parent_start
    current_ppid = snapshot.get("ppid")
    previous_ppid = ancestry.get("last_observed_ppid")
    if current_ppid != previous_ppid:
        if parent_alive or previous_ppid != parent_pid:
            raise TasteGINEControllerError(
                "trainer ancestry changed while its registered parent remained live"
            )
        ancestry["orphan_adopted"] = True
        ancestry["orphan_adopted_ppid"] = current_ppid
    ancestry["last_observed_ppid"] = current_ppid
    return {
        **dict(generation),
        "last_observed": snapshot,
        "last_observed_phase": phase,
        "phase_bindings": bindings,
        "ancestry": ancestry,
    }


def _tree_inventory(
    root: Path, *, max_files: int = 4096, max_bytes: int = 64 * 1024**3
) -> list[dict[str, Any]]:
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise TasteGINEControllerError("terminal inventory root is not physical")
    rows: list[dict[str, Any]] = []
    total = 0
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in sorted(directories):
            info = os.lstat(base / name)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteGINEControllerError("terminal tree contains symlink/special directory")
        for name in sorted(files):
            path = base / name
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                before = os.fstat(descriptor)
                named_before = os.stat(path, follow_symlinks=False)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or _file_identity(before) != _file_identity(named_before)
                ):
                    raise TasteGINEControllerError(
                        "terminal tree contains a symlink/special/aliased file"
                    )
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(descriptor)
                named_after = os.stat(path, follow_symlinks=False)
                if (
                    _file_identity(before) != _file_identity(after)
                    or _file_identity(after) != _file_identity(named_after)
                ):
                    raise TasteGINEControllerError(
                        "terminal file changed while inventoried"
                    )
            finally:
                os.close(descriptor)
            info = after
            total += int(info.st_size)
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "identity": _file_identity(info),
                    "sha256": digest.hexdigest(),
                }
            )
            if len(rows) > max_files or total > max_bytes:
                raise TasteGINEControllerError("terminal tree exceeds frozen audit limits")
    return sorted(rows, key=lambda row: row["path"])


def _published_output_adoption_evidence(
    *, output_dir: Path, training_state_root: Path
) -> dict[str, Any]:
    """Hold and rescan a published bundle whose completion write was interrupted."""

    completion_path = training_state_root / "training_complete.json"
    if completion_path.exists() or not output_dir.is_dir():
        raise TasteGINEControllerError(
            "published-output adoption requires output and missing completion"
        )
    contract_payload = _load_json(
        training_state_root / "training_contract.json",
        label="published-output training contract",
    )
    contract = contract_payload.get("contract")
    if not isinstance(contract, Mapping):
        raise TasteGINEControllerError(
            "published-output training contract content is untyped"
        )
    contract_sha = canonical_sha256(contract)
    if contract_payload.get("contract_sha256") != contract_sha:
        raise TasteGINEControllerError(
            "published-output training contract canonical SHA changed"
        )
    state_authority = MolecularGNNStateReadAuthority(
        training_state_root, contract_sha256=contract_sha
    )
    parent_authority: OutputParentAuthority | None = None
    workspace: FinalizationWorkspace | None = None
    try:
        state_authority.open()
        if completion_path.exists():
            raise TasteGINEControllerError(
                "training completion appeared before adoption authority was held"
            )
        parent_authority = OutputParentAuthority(
            output_dir,
            contract_sha256=contract_sha,
            resume=True,
            read_only=True,
        )
        parent_authority.open()
        workspace = FinalizationWorkspace(
            output_dir,
            contract_sha256=contract_sha,
            resume=True,
            parent_authority=parent_authority,
            training_state_root=training_state_root,
        )
        finalization = workspace.verify_published()
        audit = verify_checkpoint_bundle(output_dir)
        if audit["model_card"].get("training_resume_contract_sha256") != contract_sha:
            raise TasteGINEControllerError(
                "published bundle belongs to another training contract"
            )
        output_identity = _directory_identity(os.lstat(output_dir))
        state_identity = _directory_identity(os.lstat(training_state_root))
        output_inventory = _tree_inventory(output_dir)
        state_inventory = _tree_inventory(training_state_root)
        state_authority.verify()
        if completion_path.exists():
            raise TasteGINEControllerError(
                "training completion appeared during adoption scan"
            )
        if (
            output_identity != _directory_identity(os.lstat(output_dir))
            or state_identity != _directory_identity(os.lstat(training_state_root))
        ):
            raise TasteGINEControllerError(
                "published-output adoption roots changed during held scan"
            )
        return {
            "output_dir": str(output_dir.resolve(strict=True)),
            "output_identity": output_identity,
            "training_state_root": str(training_state_root.resolve(strict=True)),
            "training_state_identity": state_identity,
            "contract_sha256": contract_sha,
            "training_contract_evidence": state_authority.contract_evidence,
            "finalization": finalization,
            "output_inventory": output_inventory,
            "output_inventory_sha256": stable_sha256(output_inventory),
            "training_state_inventory": state_inventory,
            "training_state_inventory_sha256": stable_sha256(state_inventory),
            "completion_absent": True,
        }
    except (MolecularGNNResumeError, OSError, KeyError, TypeError, ValueError) as exc:
        raise TasteGINEControllerError(str(exc)) from exc
    finally:
        if workspace is not None:
            workspace.close()
        if parent_authority is not None:
            parent_authority.close()
        state_authority.close()


def validate_tastemolnet_published_output_adoption_readonly(
    receipt_path: str | Path,
    *,
    expected_output_dir: str | Path | None = None,
    expected_training_state_root: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly validate the controller-issued, completion-only resume receipt."""

    path = _absolute(receipt_path)
    if path.name != PUBLISHED_ADOPTION_NAME:
        raise TasteGINEControllerError(
            "published-output adoption receipt name is not canonical"
        )
    root = path.parent
    root_before = os.lstat(root)
    receipt = _load_json(path, label="published-output adoption receipt")
    claim = _load_json(root / ROOT_CLAIM, label="adoption controller root claim")
    spec_payload = _load_json(root / SPEC_NAME, label="adoption controller spec")
    state = _load_json(root / STATE_NAME, label="adoption controller state")
    receipt_sha = sha256_file(path)
    output = _absolute(str(receipt.get("output_dir", "")))
    training_state = _absolute(str(receipt.get("training_state_root", "")))

    def validate_live_state(value: Mapping[str, Any]) -> None:
        allowed_phases = {
            "PUBLISHED_OUTPUT_ADOPTION_PENDING",
            "ARMING",
            "STARTUP_REGISTERED",
            "RELEASE_AUTHORIZED",
            "RELEASED",
            "RUNNING",
            "RUNNING_TRAINER_ADOPTED",
        }
        deadline_path = root / RESOURCE_DEADLINE_NAME
        deadline = _load_json(
            deadline_path, label="adoption resource deadline"
        )
        state_launch_index = value.get("launch_index")
        if (
            value.get("schema_version") != STATE_SCHEMA
            or value.get("cid") != receipt.get("cid")
            or value.get("spec_sha256") != receipt.get("spec_sha256")
            or value.get("root_claim_sha256") != sha256_file(root / ROOT_CLAIM)
            or value.get("phase") not in allowed_phases
            or value.get("attempt") != receipt.get("issued_attempt")
            or not _strict_int(state_launch_index)
            or state_launch_index < receipt.get("issued_launch_index")
            or state_launch_index >= MAX_STARTUP_GENERATIONS
            or value.get("retries_used") != receipt.get("issued_attempt")
            or not isinstance(value.get("updated_at"), str)
            or value.get("resource_deadline_sha256")
            != sha256_file(deadline_path)
            or value.get("resource_deadline_epoch_seconds")
            != deadline.get("deadline_epoch_seconds")
            or value.get("published_output_adoption_receipt") != str(path)
            or value.get("published_output_adoption_sha256") != receipt_sha
        ):
            raise TasteGINEControllerError(
                "published-output adoption live state binding changed"
            )

    if (
        receipt.get("schema_version") != PUBLISHED_ADOPTION_SCHEMA
        or receipt.get("status") != "AUTHORIZED_COMPLETION_ONLY"
        or receipt.get("cid") != spec_payload.get("cid")
        or receipt.get("spec_sha256") != stable_sha256(spec_payload)
        or receipt.get("controller_root") != str(root)
        or receipt.get("output_dir") != spec_payload.get("output_dir")
        or receipt.get("training_state_root")
        != spec_payload.get("training_state_root")
        or not _strict_int(receipt.get("issued_attempt"))
        or not _strict_int(receipt.get("issued_launch_index"))
        or receipt.get("issued_attempt") < 0
        or receipt.get("issued_launch_index") < 0
        or receipt.get("issued_launch_index") >= MAX_STARTUP_GENERATIONS
        or not isinstance(receipt.get("issued_at"), str)
        or not isinstance(receipt.get("evidence"), Mapping)
        or claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("root") != str(root)
        or claim.get("root_identity") != _directory_identity(root_before)
        or claim.get("spec_sha256") != stable_sha256(spec_payload)
        or spec_payload.get("environment_authority", {}).get(
            "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT"
        )
        != str(path)
        or _file_identity(os.lstat(root / ROOT_SENTINEL))
        != claim.get("sentinel", {}).get("identity")
        or sha256_file(root / ROOT_SENTINEL)
        != claim.get("sentinel", {}).get("sha256")
        or _file_identity(os.lstat(root / ROOT_LOCK))
        != claim.get("lock", {}).get("identity")
    ):
        raise TasteGINEControllerError(
            "published-output adoption controller closure changed"
        )
    validate_live_state(state)
    if expected_output_dir is not None and output != _absolute(expected_output_dir):
        raise TasteGINEControllerError("adoption output differs from expected output")
    if (
        expected_training_state_root is not None
        and training_state != _absolute(expected_training_state_root)
    ):
        raise TasteGINEControllerError(
            "adoption training state differs from expected state"
        )
    evidence = _published_output_adoption_evidence(
        output_dir=output, training_state_root=training_state
    )
    if evidence != dict(receipt["evidence"]):
        raise TasteGINEControllerError(
            "published-output adoption sources changed"
        )
    state_after = _load_json(root / STATE_NAME, label="adoption controller state")
    validate_live_state(state_after)
    if (
        _load_json(path, label="published-output adoption receipt") != receipt
        or _directory_identity(os.lstat(root)) != _directory_identity(root_before)
    ):
        raise TasteGINEControllerError(
            "published-output adoption closure changed during read-only scan"
        )
    return receipt


class TasteGINEPersistentController:
    def __init__(self, spec: TasteGINEControllerSpec, *, resume: bool) -> None:
        self.spec = spec
        self.resume = bool(resume)
        self.spec_payload = spec.payload()
        self.spec_sha256 = stable_sha256(self.spec_payload)
        self.root = spec.controller_root
        self._root_fd: int | None = None
        self._lock_fd: int | None = None
        self._claim: dict[str, Any] | None = None
        self._terminal_readonly: dict[str, Any] | None = None

    def __enter__(self) -> "TasteGINEPersistentController":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        try:
            assert_no_symlink_components(self.root, label="Taste controller root")
        except MolecularGNNResumeError as exc:
            raise TasteGINEControllerError(str(exc)) from exc
        if self.root.exists():
            if not self.resume:
                raise TasteGINEControllerError("controller root must be fresh")
            info = os.lstat(self.root)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteGINEControllerError("controller root is not physical")
            # Terminal authority is never opened as a writer.  Detect it before
            # creating/acquiring the lock or reconciling any publication temp;
            # even a partial terminal-named artifact must fail through the one
            # shared strict, read-only validator without repair.
            existing_state: Mapping[str, Any] | None = None
            state_path = self.root / STATE_NAME
            if state_path.is_file():
                existing_state = _load_json(
                    state_path, label="pre-open controller state"
                )
            if _terminal_signal_present(self.root, existing_state):
                self._terminal_readonly = validate_tastemolnet_gine_pass_readonly(
                    self.root
                )
                return
        else:
            if self.resume:
                raise TasteGINEControllerError("resume controller root is absent")
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.root.mkdir(mode=0o700)
            _fsync_directory(self.root.parent)
        root_info = os.lstat(self.root)
        if (
            root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700
        ):
            raise TasteGINEControllerError(
                "controller root must be owner-bound mode 0700"
            )
        self._root_fd = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        sentinel = self.root / ROOT_SENTINEL
        lock = self.root / ROOT_LOCK
        claim_path = self.root / ROOT_CLAIM
        self._lock_fd = os.open(
            lock,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_info = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or lock_info.st_uid != os.getuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            self.close()
            raise TasteGINEControllerError("controller lock is not one physical file")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise TasteGINEControllerError("controller already has a live owner") from exc
        for publication_name in (
            ROOT_SENTINEL,
            ROOT_CLAIM,
            SPEC_NAME,
            STATE_NAME,
            TERMINAL_NAME,
            PASS_NAME,
        ):
            _reconcile_publication_temps(self.root, publication_name)
        if not sentinel.exists():
            allowed = {ROOT_LOCK}
            unexpected = sorted(path.name for path in self.root.iterdir() if path.name not in allowed)
            if unexpected:
                self.close()
                raise TasteGINEControllerError("unclaimed controller root is not recoverably empty")
            _write_text_new(sentinel, hashlib.sha256(os.urandom(64)).hexdigest() + "\n")
        sentinel_info = os.lstat(sentinel)
        if (
            not stat.S_ISREG(sentinel_info.st_mode)
            or sentinel_info.st_nlink != 1
            or sentinel_info.st_uid != os.getuid()
            or stat.S_IMODE(sentinel_info.st_mode) != 0o600
        ):
            self.close()
            raise TasteGINEControllerError(
                "controller sentinel must be one owner-bound mode-0600 file"
            )
        if claim_path.exists():
            claim = _load_json(claim_path, label="controller root claim")
        else:
            allowed = {ROOT_LOCK, ROOT_SENTINEL}
            unexpected = sorted(path.name for path in self.root.iterdir() if path.name not in allowed)
            if unexpected:
                self.close()
                raise TasteGINEControllerError("unclaimed controller root contains artifacts")
            claim = {
                "schema_version": CLAIM_SCHEMA,
                "cid": self.spec.cid,
                "spec_sha256": self.spec_sha256,
                "root": str(self.root),
                "root_identity": _directory_identity(os.fstat(self._root_fd)),
                "sentinel": {
                    "identity": _file_identity(sentinel_info),
                    "sha256": sha256_file(sentinel),
                },
                "lock": {"identity": _file_identity(lock_info)},
            }
            _write_json_new(claim_path, claim)
        self._claim = claim
        self.verify_authority()
        spec_path = self.root / SPEC_NAME
        _reconcile_publication_temps(self.root, SPEC_NAME)
        if spec_path.exists():
            if _load_json(spec_path, label="controller spec") != self.spec_payload:
                raise TasteGINEControllerError("controller spec changed")
        else:
            unexpected = sorted(
                path.name
                for path in self.root.iterdir()
                if path.name not in {ROOT_LOCK, ROOT_SENTINEL, ROOT_CLAIM}
            )
            if unexpected:
                raise TasteGINEControllerError("controller spec is absent after execution began")
            _write_json_new(spec_path, self.spec_payload)
        self.verify_authority()
        self._validate_spec_sources()

    def close(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def verify_authority(self) -> None:
        if self._root_fd is None or self._lock_fd is None or self._claim is None:
            raise TasteGINEControllerError("controller authority is not open")
        current = os.lstat(self.root)
        held = os.fstat(self._root_fd)
        if (
            (current.st_dev, current.st_ino) != (held.st_dev, held.st_ino)
            or _directory_identity(held) != self._claim.get("root_identity")
            or self._claim.get("schema_version") != CLAIM_SCHEMA
            or self._claim.get("cid") != self.spec.cid
            or self._claim.get("spec_sha256") != self.spec_sha256
        ):
            raise TasteGINEControllerError("controller root identity changed")
        sentinel = self.root / ROOT_SENTINEL
        sentinel_info = os.lstat(sentinel)
        if (
            _file_identity(sentinel_info) != self._claim.get("sentinel", {}).get("identity")
            or sha256_file(sentinel) != self._claim.get("sentinel", {}).get("sha256")
        ):
            raise TasteGINEControllerError("controller sentinel changed")
        lock_path = self.root / ROOT_LOCK
        lock_now = os.lstat(lock_path)
        lock_held = os.fstat(self._lock_fd)
        if (
            (lock_now.st_dev, lock_now.st_ino) != (lock_held.st_dev, lock_held.st_ino)
            or _file_identity(lock_held) != self._claim.get("lock", {}).get("identity")
        ):
            raise TasteGINEControllerError("controller lock changed")
        if _load_json(self.root / ROOT_CLAIM, label="controller root claim") != self._claim:
            raise TasteGINEControllerError("controller root claim changed")

    def _state(self) -> dict[str, Any] | None:
        path = self.root / STATE_NAME
        return None if not path.exists() else _load_json(path, label="controller state")

    def _resource_deadline(self) -> dict[str, Any]:
        """Create or reopen the one invocation-wide resource-wait deadline."""

        path = self.root / RESOURCE_DEADLINE_NAME
        if path.exists():
            payload = _load_json(path, label="resource-wait deadline")
        else:
            started = int(time.time())
            payload = {
                "schema_version": RESOURCE_DEADLINE_SCHEMA,
                "cid": self.spec.cid,
                "spec_sha256": self.spec_sha256,
                "duration_seconds": int(self.spec.resource_wait_deadline_seconds),
                "started_epoch_seconds": started,
                "deadline_epoch_seconds": started
                + int(self.spec.resource_wait_deadline_seconds),
            }
            _write_json_new(path, payload)
        if (
            payload.get("schema_version") != RESOURCE_DEADLINE_SCHEMA
            or payload.get("cid") != self.spec.cid
            or payload.get("spec_sha256") != self.spec_sha256
            or payload.get("duration_seconds")
            != int(self.spec.resource_wait_deadline_seconds)
            or not isinstance(payload.get("started_epoch_seconds"), int)
            or not isinstance(payload.get("deadline_epoch_seconds"), int)
            or payload["deadline_epoch_seconds"]
            - payload["started_epoch_seconds"]
            != int(self.spec.resource_wait_deadline_seconds)
        ):
            raise TasteGINEControllerError("resource-wait deadline authority changed")
        return payload

    def _write_state(self, phase: str, **fields: Any) -> dict[str, Any]:
        self.verify_authority()
        deadline = self._resource_deadline()
        adoption_path = self.root / PUBLISHED_ADOPTION_NAME
        adoption_fields: dict[str, Any] = {}
        if adoption_path.exists():
            adoption = _load_json(
                adoption_path, label="published-output adoption receipt"
            )
            if (
                adoption.get("schema_version") != PUBLISHED_ADOPTION_SCHEMA
                or adoption.get("cid") != self.spec.cid
                or adoption.get("spec_sha256") != self.spec_sha256
            ):
                raise TasteGINEControllerError(
                    "published-output adoption receipt changed"
                )
            adoption_fields = {
                "published_output_adoption_receipt": str(adoption_path),
                "published_output_adoption_sha256": sha256_file(adoption_path),
            }
        payload = {
            "schema_version": STATE_SCHEMA,
            "cid": self.spec.cid,
            "spec_sha256": self.spec_sha256,
            "root_claim_sha256": sha256_file(self.root / ROOT_CLAIM),
            "phase": phase,
            "updated_at": utc_now(),
            "resource_deadline_sha256": sha256_file(
                self.root / RESOURCE_DEADLINE_NAME
            ),
            "resource_deadline_epoch_seconds": deadline[
                "deadline_epoch_seconds"
            ],
            **adoption_fields,
            **fields,
        }
        _write_json_replace(self.root / STATE_NAME, payload)
        event = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        events_path = self.root / EVENTS_NAME
        if events_path.exists() or events_path.is_symlink():
            events_info = os.lstat(events_path)
            if (
                not stat.S_ISREG(events_info.st_mode)
                or events_info.st_nlink != 1
                or events_info.st_uid != os.getuid()
                or stat.S_IMODE(events_info.st_mode) != 0o600
            ):
                raise TasteGINEControllerError(
                    "controller event log is not one owner-bound file"
                )
            current_size = int(events_info.st_size)
        else:
            current_size = 0
        event_limit = (
            MAX_EVENTS_BYTES
            if phase in {"PASS", "FAILED"}
            else MAX_EVENTS_BYTES - TERMINAL_EVENT_RESERVE_BYTES
        )
        if current_size + len(event) <= event_limit:
            descriptor = os.open(
                events_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise TasteGINEControllerError(
                        "controller event log is not one owner-bound file"
                    )
                named = os.stat(events_path, follow_symlinks=False)
                if (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino):
                    raise TasteGINEControllerError(
                        "controller event log changed before append"
                    )
                offset = 0
                while offset < len(event):
                    written = os.write(descriptor, event[offset:])
                    if written <= 0:
                        raise TasteGINEControllerError(
                            "controller event log append was incomplete"
                        )
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.verify_authority()
        return payload

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema_version") != STATE_SCHEMA
            or state.get("cid") != self.spec.cid
            or state.get("spec_sha256") != self.spec_sha256
            or state.get("root_claim_sha256") != sha256_file(self.root / ROOT_CLAIM)
            or state.get("resource_deadline_sha256")
            != sha256_file(self.root / RESOURCE_DEADLINE_NAME)
            or state.get("resource_deadline_epoch_seconds")
            != self._resource_deadline()["deadline_epoch_seconds"]
        ):
            raise TasteGINEControllerError("persistent controller state changed")

    def _barrier_paths(self, launch_index: int) -> tuple[Path, Path]:
        return (
            self.root / f"startup-launch-{launch_index}.lock",
            self.root / f"startup-launch-{launch_index}.json",
        )

    def _child_environment(self) -> dict[str, str]:
        self._validate_spec_sources()
        environment = {
            key: value
            for key, value in self.spec.environment_authority.items()
            if key in FROZEN_SCIENCE_ENV_KEYS or key in FROZEN_BASE_ENV_KEYS
        }
        if self._root_fd is not None:
            deadline = self._resource_deadline()
            remaining = max(
                0,
                int(deadline["deadline_epoch_seconds"]) - int(time.time()),
            )
            environment["TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS"] = str(
                remaining
            )
            environment["TASTEMOLNET_RESOURCE_WAIT_DEADLINE_EPOCH"] = str(
                deadline["deadline_epoch_seconds"]
            )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    def _validate_spec_sources(self) -> None:
        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", "-C", str(self.spec.project_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        source = self.spec.source_identity
        if (
            git("status", "--porcelain=v1", "--untracked-files=all")
            or git("rev-parse", "HEAD") != source.get("commit")
            or git("rev-parse", "HEAD^{tree}") != source.get("tree")
            or sha256_file(str(source.get("worker_wrapper_path")))
            != source.get("worker_wrapper_sha256")
            or sha256_file(str(source.get("worker_program_path")))
            != source.get("worker_program_sha256")
            or sha256_file(Path(__file__).resolve(strict=True))
            != source.get("controller_module_sha256")
            or str(Path(sys.executable).resolve(strict=True))
            != source.get("python_executable")
            or sha256_file(Path(sys.executable).resolve(strict=True))
            != source.get("python_executable_sha256")
            or source.get("verified_backbone_config_path")
            != str((self.spec.project_root / "configs/gnn/gine.yaml").resolve(strict=True))
            or sha256_file(str(source.get("verified_backbone_config_path")))
            != source.get("verified_backbone_config_sha256")
        ):
            raise TasteGINEControllerError("immutable controller source identity changed")
        for row in self.spec.config_files:
            if sha256_file(str(row["path"])) != row.get("sha256"):
                raise TasteGINEControllerError("frozen controller config changed")
        environment = self.spec.environment_authority
        try:
            minimum_free_gb = int(environment.get("MIN_PERSISTENT_FREE_GB", "-1"))
            minimum_after_reservations_gb = int(
                environment.get("MIN_FREE_AFTER_RESERVATIONS_GB", "-1")
            )
        except (TypeError, ValueError) as exc:
            raise TasteGINEControllerError(
                "frozen persistent free-space threshold is malformed"
            ) from exc
        if (
            environment.get("PRIMARY_GNN_BACKBONE") != "gine"
            or environment.get("PRIMARY_SEED") != "7"
            or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or environment.get("PYTHONHASHSEED") != "7"
            or environment.get("NVIDIA_TF32_OVERRIDE") != "0"
            or environment.get("CUDNN_DETERMINISTIC") != "1"
            or environment.get("TASTEMOLNET_GPU_INDEX") != "1"
            or minimum_free_gb < 100
            or minimum_after_reservations_gb < 100
        ):
            raise TasteGINEControllerError(
                "frozen GINE/seed/free-space/determinism route changed"
            )
        policy = load_tastemolnet_research_policy(
            environment["TASTEMOLNET_POLICY_FILE"],
            expected_file_sha256=environment["TASTEMOLNET_POLICY_SHA256"],
        )
        policy.require_main_route()
        authority = validate_tastemolnet_local_authority(
            policy,
            prepared_root=environment["TASTEMOLNET_PREPARED_ROOT"],
            graph_cache_root=environment["TASTEMOLNET_GRAPH_CACHE_ROOT"],
        )
        receipt = validate_tastemolnet_policy_receipt(
            environment["TASTEMOLNET_POLICY_RECEIPT"],
            policy=policy,
            authority=authority,
            require_active=True,
            require_policy_version=2,
        )
        if (
            policy.canonical_sha256 != environment.get("policy_canonical_sha256")
            or receipt.sha256 != environment.get("policy_receipt_sha256")
            or stable_sha256(authority.evidence())
            != environment.get("private_authority_sha256")
        ):
            raise TasteGINEControllerError("frozen Taste policy authority changed")

    def _launch(
        self,
        attempt: int,
        *,
        launch_index: int,
        rearm: StartupBarrierRecord | None = None,
        arming_already_durable: bool = False,
    ) -> subprocess.Popen[Any]:
        if launch_index < 0 or launch_index >= MAX_STARTUP_GENERATIONS:
            raise TasteGINEControllerError("startup-generation hard cap reached")
        lock_path, record_path = self._barrier_paths(launch_index)
        if not arming_already_durable:
            self._write_state(
                "ARMING",
                attempt=attempt,
                launch_index=launch_index,
                barrier_lock=str(lock_path),
                barrier_record_path=str(record_path),
                retries_used=attempt,
            )
        barrier = arm_exec_startup_barrier(
            lock_path=lock_path,
            record_path=record_path,
            target_argv=self.spec.worker_argv,
            python_executable=sys.executable,
            record_policy="resume" if rearm is not None else "fresh",
            expected_unreleased_record=rearm,
            rearm_timeout_seconds=5.0 if rearm is not None else 0.0,
        )
        self._write_state(
            "PRE_RELEASE",
            attempt=attempt,
            launch_index=launch_index,
            barrier_record=barrier.record.to_dict(),
            worker_generation=None,
            retries_used=attempt,
        )
        log_path = self.root / f"worker-attempt-{attempt}.log"
        log_fd = os.open(
            log_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        log_info = os.fstat(log_fd)
        if (
            not stat.S_ISREG(log_info.st_mode)
            or log_info.st_nlink != 1
            or log_info.st_uid != os.getuid()
            or stat.S_IMODE(log_info.st_mode) != 0o600
            or _file_binding(os.lstat(log_path)) != _file_binding(log_info)
        ):
            os.close(log_fd)
            barrier.abort()
            raise TasteGINEControllerError(
                "worker log is not one owner-bound mode-0600 file"
            )
        log_handle = os.fdopen(log_fd, "ab", buffering=0)
        try:
            process = barrier.launch(
                cwd=self.spec.project_root,
                env=self._child_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        try:
            generation = _process_generation(
                process.pid,
                spec=self.spec,
                barrier_record=barrier.record.to_dict(),
            )
        except BaseException:
            barrier.abort()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The target was never released; preserve fail-closed ownership
                # evidence without signalling even a malformed launcher.
                pass
            raise
        common = {
            "attempt": attempt,
            "launch_index": launch_index,
            "barrier_record": barrier.record.to_dict(),
            "worker_generation": generation,
            "worker_log": str(log_path),
            "worker_log_binding": _file_binding(log_info),
            "retries_used": attempt,
        }
        self._write_state("STARTUP_REGISTERED", **common)
        self._write_state("RELEASE_AUTHORIZED", **common)
        barrier.release()
        self._write_state("RELEASED", **common)
        self._write_state("RUNNING", **common)
        return process

    def _acquire_terminal_hold(self) -> dict[str, Any] | None:
        output = self.spec.output_dir
        state_root = self.spec.training_state_root
        if not output.is_dir() or output.is_symlink() or not state_root.is_dir():
            return None

        completion_path = state_root / "training_complete.json"
        if not completion_path.is_file():
            return None
        try:
            completion = _load_json(completion_path, label="training completion")
            contract_sha = str(completion.get("contract_sha256") or "")
            if (
                completion.get("status") != "PASS"
                or completion.get("output_dir") != str(output.resolve(strict=True))
                or len(contract_sha) != 64
            ):
                return None
            state_authority = MolecularGNNStateReadAuthority(
                state_root, contract_sha256=contract_sha
            )
            state_authority.open()
            if completion.get("training_contract_evidence") != (
                state_authority.contract_evidence
            ):
                raise MolecularGNNResumeError(
                    "training completion lost physical contract evidence"
                )
            if _load_json(
                completion_path, label="training completion under state authority"
            ) != completion:
                raise MolecularGNNResumeError(
                    "training completion changed before terminal authority was held"
                )
            parent_authority = OutputParentAuthority(
                output,
                contract_sha256=contract_sha,
                resume=True,
                read_only=True,
            )
            parent_authority.open()
            workspace = FinalizationWorkspace(
                output,
                contract_sha256=contract_sha,
                resume=True,
                parent_authority=parent_authority,
                training_state_root=state_root,
            )
            workspace.verify_published()
            return {
                "completion": completion,
                "completion_path": completion_path,
                "contract_sha256": contract_sha,
                "state_authority": state_authority,
                "parent_authority": parent_authority,
                "workspace": workspace,
            }
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            ValueError,
            MolecularGNNResumeError,
            TasteResearchPolicyError,
        ):
            if "workspace" in locals():
                workspace.close()
            if "parent_authority" in locals():
                parent_authority.close()
            if "state_authority" in locals():
                state_authority.close()
            return None

    def _issue_published_output_adoption(
        self, *, attempt: int, launch_index: int
    ) -> Path:
        """Authorize only completion recovery for an already-published bundle."""

        self.verify_authority()
        path = self.root / PUBLISHED_ADOPTION_NAME
        if path.exists():
            # Receipt publication is no-clobber and may durably win just before
            # the controller state replacement.  Reopen the immutable receipt
            # and all held science sources first, then repair only that
            # nonterminal state-link window under the same controller lock.
            receipt = _load_json(
                path, label="interrupted published-output adoption receipt"
            )
            evidence = _published_output_adoption_evidence(
                output_dir=self.spec.output_dir,
                training_state_root=self.spec.training_state_root,
            )
            if (
                receipt.get("schema_version") != PUBLISHED_ADOPTION_SCHEMA
                or receipt.get("status") != "AUTHORIZED_COMPLETION_ONLY"
                or receipt.get("cid") != self.spec.cid
                or receipt.get("spec_sha256") != self.spec_sha256
                or receipt.get("controller_root") != str(self.root)
                or receipt.get("output_dir") != str(self.spec.output_dir)
                or receipt.get("training_state_root")
                != str(self.spec.training_state_root)
                or not _strict_int(receipt.get("issued_attempt"))
                or not _strict_int(receipt.get("issued_launch_index"))
                or receipt.get("issued_attempt") < 0
                or receipt.get("issued_launch_index") < 0
                or receipt.get("issued_attempt") != attempt
                or receipt.get("issued_launch_index") > launch_index
                or launch_index >= MAX_STARTUP_GENERATIONS
                or not isinstance(receipt.get("issued_at"), str)
                or receipt.get("evidence") != evidence
                or self.spec.environment_authority.get(
                    "TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT"
                )
                != str(path)
            ):
                raise TasteGINEControllerError(
                    "interrupted published-output adoption receipt changed"
                )
            self.verify_authority()
            state = self._state()
            receipt_sha = sha256_file(path)
            if (
                not isinstance(state, Mapping)
                or state.get("published_output_adoption_receipt") != str(path)
                or state.get("published_output_adoption_sha256") != receipt_sha
                or state.get("launch_index") != launch_index
                or state.get("phase") != "PUBLISHED_OUTPUT_ADOPTION_PENDING"
            ):
                self._write_state(
                    "PUBLISHED_OUTPUT_ADOPTION_PENDING",
                    attempt=attempt,
                    launch_index=launch_index,
                    retries_used=attempt,
                    reason=(
                        "RECOVERED_RECEIPT_BEFORE_STATE_CRASH_WINDOW"
                    ),
                )
            validate_tastemolnet_published_output_adoption_readonly(
                path,
                expected_output_dir=self.spec.output_dir,
                expected_training_state_root=self.spec.training_state_root,
            )
            return path
        evidence = _published_output_adoption_evidence(
            output_dir=self.spec.output_dir,
            training_state_root=self.spec.training_state_root,
        )
        payload = {
            "schema_version": PUBLISHED_ADOPTION_SCHEMA,
            "status": "AUTHORIZED_COMPLETION_ONLY",
            "cid": self.spec.cid,
            "spec_sha256": self.spec_sha256,
            "controller_root": str(self.root),
            "output_dir": str(self.spec.output_dir),
            "training_state_root": str(self.spec.training_state_root),
            "issued_attempt": int(attempt),
            "issued_launch_index": int(launch_index),
            "issued_at": utc_now(),
            "evidence": evidence,
        }
        _write_json_new(path, payload)
        self._write_state(
            "PUBLISHED_OUTPUT_ADOPTION_PENDING",
            attempt=attempt,
            launch_index=launch_index,
            retries_used=attempt,
            reason="FINALIZATION_PUBLISHED_COMPLETION_WRITE_INTERRUPTED",
        )
        validate_tastemolnet_published_output_adoption_readonly(
            path,
            expected_output_dir=self.spec.output_dir,
            expected_training_state_root=self.spec.training_state_root,
        )
        return path

    def _discover_trainer_generation(
        self, state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Find exactly one live, durably registered trainer descendant."""

        raw_control_root = self.spec.environment_authority.get("AUTODL_CONTROL_ROOT")
        worker_generation = state.get("worker_generation")
        worker_barrier = state.get("barrier_record")
        if (
            not raw_control_root
            or not isinstance(worker_generation, Mapping)
            or not isinstance(worker_barrier, Mapping)
        ):
            return None
        runs_root = _absolute(raw_control_root) / "experiment_registry/run_state"
        candidates: set[Path] = set()
        stored = state.get("trainer_generation")
        if isinstance(stored, Mapping) and isinstance(
            stored.get("authority_path"), str
        ):
            candidates.add(Path(str(stored["authority_path"])))
        if runs_root.is_dir():
            discovered = sorted(runs_root.glob(f"*/{TRAINER_CHILD_AUTHORITY_NAME}"))
            if len(discovered) > MAX_STARTUP_GENERATIONS:
                raise TasteGINEControllerError(
                    "trainer authority discovery exceeded its bounded generation cap"
                )
            for path in discovered:
                raw = _load_json(path, label="candidate trainer child authority")
                if (
                    raw.get("controller_cid") == self.spec.cid
                    or raw.get("controller_root") == str(self.root)
                ):
                    if (
                        raw.get("controller_cid") != self.spec.cid
                        or raw.get("controller_root") != str(self.root)
                    ):
                        raise TasteGINEControllerError(
                            "trainer authority partially collides with controller identity"
                        )
                    candidates.add(path)
        live: list[dict[str, Any]] = []
        for path in sorted(candidates):
            authority, authority_identity, authority_sha256 = (
                _load_trainer_child_authority_structure(path, spec=self.spec)
            )
            child = authority["child_registered"]
            declared_live = _declared_trainer_child_is_live(child)
            _verify_trainer_child_authority_evidence(
                path,
                identity=authority_identity,
                sha256=authority_sha256,
            )
            if not declared_live:
                # A structurally valid authority for a PID/start pair that no
                # longer exists is historical.  It must not be rebound to the
                # current worker, but malformed or unreadable evidence above
                # remains a hard failure.
                continue
            base = _trainer_generation_from_authority(
                path,
                spec=self.spec,
                worker_generation=worker_generation,
                worker_barrier_record=worker_barrier,
            )
            candidate = base
            if (
                isinstance(stored, Mapping)
                and stored.get("authority_path") == str(path)
            ):
                stable_keys = (
                    "pid",
                    "linux_start_ticks",
                    "registered",
                    "registered_phase",
                    "authority_path",
                    "authority_sha256",
                    "authority_identity",
                    "barrier_record",
                )
                if any(stored.get(key) != base.get(key) for key in stable_keys):
                    raise TasteGINEControllerError(
                        "persisted trainer child registration changed"
                    )
                candidate = dict(stored)
            observed = _observe_trainer_generation(candidate)
            if observed is not None:
                live.append(observed)
        if len(live) > 1:
            raise TasteGINEControllerError(
                "multiple live trainer generations forbid concurrent retry"
            )
        return None if not live else live[0]

    def _release_terminal_hold(self, hold: Mapping[str, Any]) -> None:
        workspace = hold["workspace"]
        parent_authority = hold["parent_authority"]
        state_authority = hold["state_authority"]
        workspace.close()
        parent_authority.close()
        state_authority.close()

    def _scan_terminal_hold(self, hold: Mapping[str, Any]) -> dict[str, Any] | None:
        output = self.spec.output_dir
        state_root = self.spec.training_state_root
        completion = hold["completion"]
        completion_path = hold["completion_path"]
        contract_sha = str(hold["contract_sha256"])
        workspace = hold["workspace"]
        state_authority = hold["state_authority"]
        try:
            state_authority.verify()
            if _load_json(
                completion_path, label="held training completion"
            ) != completion:
                raise TasteGINEControllerError(
                    "training completion changed while terminal authority was held"
                )
            finalization_evidence = workspace.verify_published()
            audit = verify_checkpoint_bundle(output)
            model_card = audit["model_card"]
            if contract_sha != model_card.get("training_resume_contract_sha256"):
                return None
            completion_identity = {
                "model_sha256": sha256_file(output / "model.pt"),
                "model_card_sha256": sha256_file(output / "model_card.json"),
                "sha256s_sha256": sha256_file(output / "sha256sums.txt"),
                "checkpoint_id": model_card.get("checkpoint_id"),
                "training_resume_contract_sha256": contract_sha,
                "finalization_claim_sha256": finalization_evidence["claim_sha256"],
                "finalization_completion_sha256": finalization_evidence[
                    "completion_sha256"
                ],
            }
            if completion.get("output_identity") != completion_identity:
                return None
            policy_binding = _load_json(
                output / "data_use_policy_binding.json", label="Taste policy binding"
            )
            if (
                policy_binding.get("status") != "NOT_EXPLICITLY_STATED"
                or policy_binding.get("authorization_status")
                != "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
                or policy_binding.get("dataset_redistributed") is not False
            ):
                return None
            policy_evidence = policy_binding.get("policy")
            private_evidence = policy_binding.get("private_data_authority")
            receipt_evidence = policy_binding.get("policy_receipt")
            if not all(
                isinstance(value, Mapping)
                for value in (policy_evidence, private_evidence, receipt_evidence)
            ):
                return None
            policy = load_tastemolnet_research_policy(
                str(policy_evidence["policy_path"]),
                expected_file_sha256=str(policy_evidence["policy_file_sha256"]),
            )
            authority = validate_tastemolnet_local_authority(
                policy,
                prepared_root=str(private_evidence["prepared_root"]),
                graph_cache_root=str(private_evidence["graph_cache_root"]),
            )
            receipt = validate_tastemolnet_policy_receipt(
                str(receipt_evidence["path"]),
                policy=policy,
                authority=authority,
                require_active=True,
                require_policy_version=2,
            )
            if (
                policy.evidence() != policy_evidence
                or authority.evidence() != private_evidence
                or receipt.sha256 != receipt_evidence.get("sha256")
            ):
                return None
            output_before = _directory_identity(os.lstat(output))
            state_before = _directory_identity(os.lstat(state_root))
            output_inventory = _tree_inventory(output)
            state_inventory = _tree_inventory(state_root)
            output_after = _directory_identity(os.lstat(output))
            state_after = _directory_identity(os.lstat(state_root))
            if output_before != output_after or state_before != state_after:
                raise TasteGINEControllerError("terminal root identity changed during scan")
            state_authority.verify()
            return {
                "output_dir": str(output.resolve(strict=True)),
                "output_identity": output_before,
                "training_state_root": str(state_root.resolve(strict=True)),
                "training_state_identity": state_before,
                "contract_sha256": contract_sha,
                "training_contract_evidence": state_authority.contract_evidence,
                "completion_sha256": sha256_file(completion_path),
                "finalization": finalization_evidence,
                "output_inventory": output_inventory,
                "output_inventory_sha256": stable_sha256(output_inventory),
                "training_state_inventory": state_inventory,
                "training_state_inventory_sha256": stable_sha256(state_inventory),
                "policy_file_sha256": policy.file_sha256,
                "policy_receipt_sha256": receipt.sha256,
            }
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            ValueError,
            MolecularGNNResumeError,
            TasteResearchPolicyError,
        ):
            return None
    def _terminal_evidence(self) -> dict[str, Any] | None:
        hold = self._acquire_terminal_hold()
        if hold is None:
            return None
        try:
            return self._scan_terminal_hold(hold)
        finally:
            self._release_terminal_hold(hold)

    def _publish_terminal(
        self, evidence: Mapping[str, Any], *, attempt: int, launch_index: int
    ) -> None:
        self.verify_authority()
        hold = self._acquire_terminal_hold()
        if hold is None:
            raise TasteGINEControllerError("terminal sources are not quiescent")
        marker_identity: dict[str, int] | None = None
        try:
            self._validate_spec_sources()
            if self._scan_terminal_hold(hold) != dict(evidence):
                raise TasteGINEControllerError("terminal bundle changed before held scan")
            if self.spec.terminal_stability_seconds:
                time.sleep(self.spec.terminal_stability_seconds)
            if self._scan_terminal_hold(hold) != dict(evidence):
                raise TasteGINEControllerError("terminal bundle changed during held scan")
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": "PASS",
                "cid": self.spec.cid,
                "spec_sha256": self.spec_sha256,
                "attempt": attempt,
                "launch_index": launch_index,
                "process_loss_retries_used": attempt,
                "same_training_state_root": str(self.spec.training_state_root),
                "evidence": dict(evidence),
            }
            terminal_path = self.root / TERMINAL_NAME
            if terminal_path.exists():
                if _load_json(terminal_path, label="controller terminal") != terminal:
                    raise TasteGINEControllerError("controller terminal changed")
            else:
                _write_json_new(terminal_path, terminal)
            final_state = self._write_state(
                "PASS",
                attempt=attempt,
                launch_index=launch_index,
                retries_used=attempt,
                terminal_sha256=sha256_file(terminal_path),
            )
            if self._scan_terminal_hold(hold) != dict(evidence):
                raise TasteGINEControllerError("terminal sources changed before PASS publication")
            if _load_json(self.root / STATE_NAME, label="controller final state") != final_state:
                raise TasteGINEControllerError("controller final state changed before PASS")
            if _load_json(self.root / SPEC_NAME, label="controller spec") != self.spec_payload:
                raise TasteGINEControllerError("controller spec changed before PASS")
            self._validate_spec_sources()
            if not (self.root / PASS_NAME).exists():
                _write_text_new(self.root / PASS_NAME, "TASTEMOLNET_GINE_CONTROLLER_PASS\n")
            marker_text, marker_identity = _load_text_bound(
                self.root / PASS_NAME, label="controller PASS marker"
            )
            if marker_text != "TASTEMOLNET_GINE_CONTROLLER_PASS\n":
                raise TasteGINEControllerError("controller PASS marker content changed")
            # PASS is not the end of validation: retain controller/output/state
            # locks and reopen every source once more after the marker is named.
            if self._scan_terminal_hold(hold) != dict(evidence):
                raise TasteGINEControllerError(
                    "terminal sources changed after PASS publication"
                )
            if _load_json(terminal_path, label="controller terminal after PASS") != terminal:
                raise TasteGINEControllerError(
                    "controller terminal changed after PASS publication"
                )
            if _load_json(self.root / STATE_NAME, label="controller state after PASS") != final_state:
                raise TasteGINEControllerError(
                    "controller state changed after PASS publication"
                )
            if _load_json(self.root / SPEC_NAME, label="controller spec after PASS") != self.spec_payload:
                raise TasteGINEControllerError(
                    "controller spec changed after PASS publication"
                )
            self._validate_spec_sources()
            self.verify_authority()
        except BaseException:
            # Revoke only the exact marker inode this publication observed.  A
            # concurrently substituted name is never unlinked by pathname.
            marker_path = self.root / PASS_NAME
            if marker_identity is not None:
                try:
                    current = os.lstat(marker_path)
                except FileNotFoundError:
                    pass
                else:
                    if (
                        int(current.st_dev) == marker_identity["device"]
                        and int(current.st_ino) == marker_identity["inode"]
                    ):
                        marker_path.unlink()
                        _fsync_directory(self.root)
            raise
        finally:
            self._release_terminal_hold(hold)

    def _failure_is_semantic(self, state: Mapping[str, Any], returncode: int | None) -> bool:
        if returncode is not None and returncode >= 0:
            return True
        log_value = state.get("worker_log")
        if isinstance(log_value, str) and Path(log_value).is_file():
            tail = Path(log_value).read_bytes()[-128 * 1024 :]
            if b"AUTODL_RUN_FAILED" in tail or b"HEALTH_GATE_FAILED" in tail:
                return True
        return False

    def _resource_deadline_expired(self) -> bool:
        return int(time.time()) >= int(
            self._resource_deadline()["deadline_epoch_seconds"]
        )

    def _bound_worker_log_size(
        self, state: Mapping[str, Any], *, truncate_to_cap: bool
    ) -> int:
        raw_path = state.get("worker_log")
        binding = state.get("worker_log_binding")
        attempt = int(state.get("attempt", -1))
        expected_path = self.root / f"worker-attempt-{attempt}.log"
        if (
            not isinstance(raw_path, str)
            or Path(raw_path) != expected_path
            or not isinstance(binding, Mapping)
        ):
            raise TasteGINEControllerError("worker log authority is absent")
        flags = os.O_RDWR if truncate_to_cap else os.O_RDONLY
        descriptor = os.open(
            expected_path, flags | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            info = os.fstat(descriptor)
            named = os.stat(expected_path, follow_symlinks=False)
            if (
                _file_binding(info) != dict(binding)
                or _file_binding(named) != dict(binding)
                or not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise TasteGINEControllerError("worker log physical binding changed")
            if truncate_to_cap and info.st_size > MAX_WORKER_LOG_BYTES:
                os.ftruncate(descriptor, MAX_WORKER_LOG_BYTES)
                os.fsync(descriptor)
                info = os.fstat(descriptor)
            return int(info.st_size)
        finally:
            os.close(descriptor)

    def _supervise_identity_drift(
        self,
        state: Mapping[str, Any],
        *,
        process: subprocess.Popen[Any] | None,
        reason: str,
    ) -> int:
        """Retain ownership without signalling an identity-drifted live PID."""

        generation = state.get("worker_generation")
        if not isinstance(generation, Mapping):
            raise TasteGINEControllerError("identity drift lacks worker generation")
        self._write_state(
            "SUPERVISING_IDENTITY_DRIFT",
            attempt=int(state.get("attempt", 0)),
            launch_index=int(state.get("launch_index", 0)),
            retries_used=int(state.get("attempt", 0)),
            worker_generation=dict(generation),
            worker_log=state.get("worker_log"),
            worker_log_binding=state.get("worker_log_binding"),
            reason=reason,
        )
        while True:
            alive = (
                process.poll() is None
                if process is not None
                else _generation_alive(generation)
            )
            if not alive:
                self._write_state(
                    "FAILED",
                    attempt=int(state.get("attempt", 0)),
                    launch_index=int(state.get("launch_index", 0)),
                    retries_used=int(state.get("attempt", 0)),
                    reason="WORKER_PROCESS_IDENTITY_DRIFT",
                )
                return 2
            time.sleep(self.spec.poll_seconds)

    def _supervise_trainer_identity_drift(
        self,
        state: Mapping[str, Any],
        *,
        process: subprocess.Popen[Any] | None,
        reason: str,
    ) -> int:
        """Retain fail-closed ownership of every PID/start named by child authority."""

        generations: list[dict[str, Any]] = []
        unresolved_authority = False
        stored = state.get("trainer_generation")
        if isinstance(stored, Mapping):
            generations.append(dict(stored))
        elif stored is not None:
            unresolved_authority = True
        raw_control = self.spec.environment_authority.get("AUTODL_CONTROL_ROOT")
        if raw_control:
            runs_root = _absolute(raw_control) / "experiment_registry/run_state"
            try:
                paths = (
                    sorted(runs_root.glob(f"*/{TRAINER_CHILD_AUTHORITY_NAME}"))
                    if runs_root.is_dir()
                    else []
                )
            except OSError:
                paths = []
                unresolved_authority = True
            if len(paths) > MAX_STARTUP_GENERATIONS:
                unresolved_authority = True
                paths = paths[:MAX_STARTUP_GENERATIONS]
            for path in paths:
                try:
                    raw = _load_json(
                        path, label="drifted trainer child authority"
                    )
                except (TasteGINEControllerError, OSError, ValueError):
                    unresolved_authority = True
                    continue
                cid_matches = raw.get("controller_cid") == self.spec.cid
                root_matches = raw.get("controller_root") == str(self.root)
                if not cid_matches and not root_matches:
                    continue
                child = raw.get("child_registered")
                if not cid_matches or not root_matches or not isinstance(
                    child, Mapping
                ):
                    unresolved_authority = True
                    continue
                generations.append(
                    {
                        "pid": child.get("pid"),
                        "linux_start_ticks": child.get("linux_start_ticks"),
                        "authority_path": str(path),
                    }
                )
        unique: dict[tuple[int, Any], dict[str, Any]] = {}
        for value in generations:
            try:
                pid = int(value.get("pid", -1))
            except (TypeError, ValueError):
                unresolved_authority = True
                continue
            start = value.get("linux_start_ticks")
            if pid <= 0 or (
                sys.platform.startswith("linux")
                and (not _strict_int(start) or start <= 0)
            ):
                unresolved_authority = True
                continue
            unique[(pid, start)] = value
        self._write_state(
            "SUPERVISING_TRAINER_IDENTITY_DRIFT",
            attempt=int(state.get("attempt", 0)),
            launch_index=int(state.get("launch_index", 0)),
            retries_used=int(state.get("attempt", 0)),
            worker_generation=state.get("worker_generation"),
            worker_log=state.get("worker_log"),
            worker_log_binding=state.get("worker_log_binding"),
            trainer_drift_generations=list(unique.values()),
            trainer_drift_unresolved_authority=unresolved_authority,
            reason=reason,
        )
        while True:
            parent_alive = (
                process.poll() is None
                if process is not None
                else isinstance(state.get("worker_generation"), Mapping)
                and _generation_alive(state["worker_generation"])
            )
            trainer_alive = unresolved_authority or any(
                _linux_start_ticks(pid) == start
                if sys.platform.startswith("linux")
                else _generation_alive(value)
                for (pid, start), value in unique.items()
            )
            if not parent_alive and not trainer_alive:
                self._write_state(
                    "FAILED",
                    attempt=int(state.get("attempt", 0)),
                    launch_index=int(state.get("launch_index", 0)),
                    retries_used=int(state.get("attempt", 0)),
                    reason="TRAINER_PROCESS_IDENTITY_DRIFT",
                )
                return 2
            time.sleep(self.spec.poll_seconds)

    def _monitor(
        self,
        state: Mapping[str, Any],
        *,
        process: subprocess.Popen[Any] | None,
    ) -> int:
        attempt = int(state["attempt"])
        generation = state.get("worker_generation")
        if not isinstance(generation, Mapping):
            raise TasteGINEControllerError("RUNNING state lacks worker generation")
        log_cap_observed = bool(state.get("worker_log_cap_observed", False))
        trainer_generation = (
            dict(state["trainer_generation"])
            if isinstance(state.get("trainer_generation"), Mapping)
            else None
        )
        while True:
            if self._bound_worker_log_size(
                state, truncate_to_cap=True
            ) >= MAX_WORKER_LOG_BYTES:
                # The controller must never abandon ownership of a live science
                # generation merely because diagnostics reached their cap.
                log_cap_observed = True
            evidence = self._terminal_evidence()
            if evidence is not None:
                self._publish_terminal(
                    evidence,
                    attempt=attempt,
                    launch_index=int(state.get("launch_index", 0)),
                )
                return 0
            discovery_state = {
                **dict(state),
                "worker_generation": dict(generation),
                "trainer_generation": trainer_generation,
            }
            try:
                observed_trainer = self._discover_trainer_generation(
                    discovery_state
                )
            except (
                TasteGINEControllerError,
                OSError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                return self._supervise_trainer_identity_drift(
                    state,
                    process=process,
                    reason=f"trainer child identity drift: {exc}",
                )
            trainer_was_registered = trainer_generation is not None
            trainer_generation = observed_trainer
            if process is not None:
                returncode = process.poll()
                alive = returncode is None
            else:
                returncode = None
                alive = _generation_alive(generation)
            if not alive and trainer_generation is not None:
                state = self._write_state(
                    "RUNNING_TRAINER_ADOPTED",
                    attempt=attempt,
                    launch_index=int(state.get("launch_index", 0)),
                    barrier_record=state.get("barrier_record"),
                    worker_generation=dict(generation),
                    trainer_generation=dict(trainer_generation),
                    trainer_authority_path=trainer_generation.get(
                        "authority_path"
                    ),
                    worker_log=state.get("worker_log"),
                    worker_log_binding=state.get("worker_log_binding"),
                    retries_used=attempt,
                    adopted_after_exp_run_parent_loss=True,
                    worker_log_cap_observed=log_cap_observed,
                )
                # Never retry while the real trainer child remains alive.  The
                # next loop revalidates its PID/start/cwd/argv/cmd/exe/ancestry.
                time.sleep(self.spec.poll_seconds)
                continue
            if alive:
                barrier_record = state.get("barrier_record")
                if not isinstance(barrier_record, Mapping):
                    raise TasteGINEControllerError(
                        "RUNNING state lacks barrier authority"
                    )
                try:
                    observed_generation = _observe_generation(
                        generation,
                        spec=self.spec,
                        barrier_record=barrier_record,
                    )
                except (
                    TasteGINEControllerError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                ) as exc:
                    return self._supervise_identity_drift(
                        state,
                        process=process,
                        reason=str(exc),
                    )
                if observed_generation is None:
                    alive = False
                else:
                    generation = observed_generation
            if not alive:
                if trainer_was_registered:
                    # A normally exiting trainer writes completion immediately
                    # before process exit.  Give that durable publication one
                    # bounded poll before classifying the generation as lost.
                    evidence = self._terminal_evidence()
                    if evidence is None:
                        time.sleep(self.spec.poll_seconds)
                        evidence = self._terminal_evidence()
                    if evidence is not None:
                        self._publish_terminal(
                            evidence,
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                        )
                        return 0
                published_without_completion = (
                    self.spec.output_dir.is_dir()
                    and not self.spec.output_dir.is_symlink()
                    and any(self.spec.output_dir.iterdir())
                    and self.spec.training_state_root.is_dir()
                    and not (
                        self.spec.training_state_root / "training_complete.json"
                    ).exists()
                )
                if published_without_completion:
                    next_launch = int(state.get("launch_index", 0)) + 1
                    try:
                        self._issue_published_output_adoption(
                            attempt=attempt, launch_index=next_launch
                        )
                    except (TasteGINEControllerError, OSError) as exc:
                        self._write_state(
                            "FAILED",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="UNSAFE_PUBLISHED_OUTPUT_WITHOUT_COMPLETION",
                            detail=str(exc),
                        )
                        return 2
                    relaunched = self._launch(
                        attempt, launch_index=next_launch
                    )
                    return self._monitor(
                        self._state_or_error(), process=relaunched
                    )
                if returncode == 75:
                    next_launch = int(state.get("launch_index", 0)) + 1
                    if self._resource_deadline_expired():
                        self._write_state(
                            "FAILED",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="RESOURCE_WAIT_GLOBAL_DEADLINE_EXCEEDED",
                            returncode=75,
                            worker_log_cap_observed=log_cap_observed,
                        )
                        return 75
                    if next_launch >= MAX_STARTUP_GENERATIONS:
                        self._write_state(
                            "WAITING_RESOURCES",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="RESOURCE_WAIT_GENERATION_CAP_SUPERVISED",
                            resource_generation_cap=True,
                            returncode=75,
                            worker_log_cap_observed=log_cap_observed,
                        )
                        while not self._resource_deadline_expired():
                            time.sleep(self.spec.poll_seconds)
                        self._write_state(
                            "FAILED",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="RESOURCE_WAIT_STARTUP_GENERATION_CAP_EXCEEDED",
                            returncode=75,
                            worker_log_cap_observed=log_cap_observed,
                        )
                        return 75
                    self._write_state(
                        "WAITING_RESOURCES",
                        attempt=attempt,
                        launch_index=next_launch,
                        retries_used=attempt,
                        reason="TRANSIENT_RESOURCE_WAIT_EXIT_75",
                        worker_log_cap_observed=log_cap_observed,
                    )
                    time.sleep(self.spec.poll_seconds)
                    if self._resource_deadline_expired():
                        self._write_state(
                            "FAILED",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="RESOURCE_WAIT_GLOBAL_DEADLINE_EXCEEDED",
                            returncode=75,
                            worker_log_cap_observed=log_cap_observed,
                        )
                        return 75
                    relaunched = self._launch(
                        attempt,
                        launch_index=next_launch,
                    )
                    return self._monitor(
                        self._state_or_error(), process=relaunched
                    )
                if self._resource_deadline_expired():
                    log_value = state.get("worker_log")
                    tail = b""
                    if isinstance(log_value, str) and Path(log_value).is_file():
                        tail = Path(log_value).read_bytes()[-128 * 1024 :]
                    if b"TASTEMOLNET_RESOURCE_WAIT_DEADLINE_EXCEEDED" in tail:
                        self._write_state(
                            "FAILED",
                            attempt=attempt,
                            launch_index=int(state.get("launch_index", 0)),
                            retries_used=attempt,
                            reason="RESOURCE_WAIT_GLOBAL_DEADLINE_EXCEEDED",
                            returncode=returncode,
                            worker_log_cap_observed=log_cap_observed,
                        )
                        return 75
                if self._failure_is_semantic(state, returncode):
                    self._write_state(
                        "FAILED",
                        attempt=attempt,
                        launch_index=int(state.get("launch_index", 0)),
                        retries_used=attempt,
                        reason="SCIENTIFIC_OR_NORMAL_PROCESS_FAILURE",
                        returncode=returncode,
                        worker_log_cap_observed=log_cap_observed,
                    )
                    return 2
                if attempt + 1 >= self.spec.max_attempts:
                    self._write_state(
                        "FAILED",
                        attempt=attempt,
                        retries_used=attempt,
                        reason="PROCESS_LOSS_RETRY_EXHAUSTED",
                        returncode=returncode,
                        worker_log_cap_observed=log_cap_observed,
                    )
                    return 3
                self._write_state(
                    "RETRY_PENDING",
                    attempt=attempt,
                    retries_used=attempt,
                    reason="WORKER_PROCESS_GENERATION_LOST",
                    returncode=returncode,
                    worker_log_cap_observed=log_cap_observed,
                )
                retried = self._launch(
                    attempt + 1,
                    launch_index=int(state.get("launch_index", 0)) + 1,
                )
                return self._monitor(self._state_or_error(), process=retried)
            state = self._write_state(
                "RUNNING",
                attempt=attempt,
                launch_index=int(state.get("launch_index", 0)),
                barrier_record=state.get("barrier_record"),
                worker_generation=dict(generation),
                worker_log=state.get("worker_log"),
                worker_log_binding=state.get("worker_log_binding"),
                retries_used=attempt,
                heartbeat="worker_generation_alive",
                worker_log_cap_observed=log_cap_observed,
                trainer_generation=trainer_generation,
                trainer_authority_path=(
                    None
                    if trainer_generation is None
                    else trainer_generation.get("authority_path")
                ),
            )
            time.sleep(self.spec.poll_seconds)

    def _state_or_error(self) -> dict[str, Any]:
        state = self._state()
        if state is None:
            raise TasteGINEControllerError("controller state unexpectedly absent")
        self._validate_state(state)
        return state

    def _failed_identity_drift_generations_are_quiescent(self) -> bool:
        """Prove that a reviewed exp_run/trainer pair exited before adoption.

        This is deliberately narrower than generic FAILED recovery.  Every
        durable trainer authority that collides with this controller must bind
        both its CID and root, must describe the exact reviewed exp_run and
        trainer startup argv/executable phases, and both PID/start generations
        must be conclusively absent.  Any malformed, partial, live, or
        ambiguous authority keeps the controller FAILED.
        """

        raw_control_root = self.spec.environment_authority.get(
            "AUTODL_CONTROL_ROOT"
        )
        if not raw_control_root:
            return False
        runs_root = _absolute(raw_control_root) / "experiment_registry/run_state"
        if not runs_root.is_dir():
            return False
        try:
            paths = sorted(runs_root.glob(f"*/{TRAINER_CHILD_AUTHORITY_NAME}"))
        except OSError:
            return False
        matched = 0
        for path in paths:
            try:
                assert_no_symlink_components(
                    path, label="failed-run trainer child authority path"
                )
            except MolecularGNNResumeError as exc:
                raise TasteGINEControllerError(str(exc)) from exc
            raw = _load_json(path, label="failed-run trainer child authority")
            cid_matches = raw.get("controller_cid") == self.spec.cid
            root_matches = raw.get("controller_root") == str(self.root)
            if not cid_matches and not root_matches:
                continue
            if not cid_matches or not root_matches:
                raise TasteGINEControllerError(
                    "trainer authority partially collides with failed controller"
                )
            matched += 1
            if matched > MAX_STARTUP_GENERATIONS:
                raise TasteGINEControllerError(
                    "failed controller trainer authority count exceeds generation cap"
                )
            authority, identity, digest = _load_trainer_child_authority_structure(
                path, spec=self.spec
            )
            parent = authority["parent_exp_run"]
            child = authority["child_registered"]
            barrier = authority["barrier_record"]
            if (
                _classify_process_phase(
                    parent, spec=self.spec, barrier_record=barrier
                )
                != "exp_run_target"
                or _classify_trainer_phase(child, barrier_record=barrier)
                != "trainer_startup_launcher"
            ):
                raise TasteGINEControllerError(
                    "failed controller generations are not reviewed execution phases"
                )
            parent_live = _declared_trainer_child_is_live(parent)
            child_live = _declared_trainer_child_is_live(child)
            _verify_trainer_child_authority_evidence(
                path, identity=identity, sha256=digest
            )
            if parent_live or child_live:
                return False
        return matched > 0

    def run(self) -> int:
        if self._terminal_readonly is not None:
            return 0
        self.verify_authority()
        state = self._state()
        if _terminal_signal_present(self.root, state):
            validate_tastemolnet_gine_pass_readonly(
                self.root, held_controller=self
            )
            return 0
        if state is None:
            process = self._launch(0, launch_index=0)
            return self._monitor(self._state_or_error(), process=process)
        self._validate_state(state)
        phase = str(state.get("phase"))
        attempt = int(state.get("attempt", 0))
        if phase in {"FAILED"}:
            if state.get("reason") != "WORKER_PROCESS_IDENTITY_DRIFT":
                return 2
            try:
                generations_are_quiescent = (
                    self._failed_identity_drift_generations_are_quiescent()
                )
            except (
                TasteGINEControllerError,
                OSError,
                TypeError,
                ValueError,
                KeyError,
            ):
                return 2
            if not generations_are_quiescent:
                return 2
            try:
                evidence = self._terminal_evidence()
            except (
                TasteGINEControllerError,
                OSError,
                TypeError,
                ValueError,
                KeyError,
            ):
                return 2
            if evidence is None:
                return 2
            self._publish_terminal(
                evidence,
                attempt=attempt,
                launch_index=int(state.get("launch_index", 0)),
            )
            return 0
        if phase == "ARMING":
            launch_index = int(state.get("launch_index", 0))
            lock_path, record_path = self._barrier_paths(launch_index)
            exists = reconcile_interrupted_startup_barrier_publication(
                lock_path=lock_path,
                record_path=record_path,
                timeout_seconds=10.0,
            )
            if exists:
                record = validate_reopenable_unreleased_barrier(
                    record_path,
                    expected_target_argv=self.spec.worker_argv,
                    timeout_seconds=10.0,
                )
                process = self._launch(
                    attempt,
                    launch_index=launch_index,
                    rearm=record,
                    arming_already_durable=True,
                )
            else:
                process = self._launch(
                    attempt,
                    launch_index=launch_index,
                    arming_already_durable=True,
                )
            return self._monitor(self._state_or_error(), process=process)
        if phase in {"PRE_RELEASE", "STARTUP_REGISTERED"}:
            raw = state.get("barrier_record")
            if not isinstance(raw, Mapping):
                raise TasteGINEControllerError("pre-release state lost barrier authority")
            launch_index = int(state.get("launch_index", 0))
            record_path = self._barrier_paths(launch_index)[1]
            record = validate_startup_barrier_record(
                record_path, expected_target_argv=self.spec.worker_argv
            )
            if record.to_dict() != dict(raw):
                raise TasteGINEControllerError("pre-release barrier record changed")
            record = validate_reopenable_unreleased_barrier(
                record_path,
                expected_target_argv=self.spec.worker_argv,
                timeout_seconds=10.0,
            )
            process = self._launch(
                attempt,
                launch_index=launch_index,
                rearm=record,
                arming_already_durable=True,
            )
            return self._monitor(self._state_or_error(), process=process)
        if phase == "RELEASE_AUTHORIZED":
            generation = state.get("worker_generation")
            raw = state.get("barrier_record")
            if not isinstance(raw, Mapping):
                raise TasteGINEControllerError(
                    "release-authorized state lost barrier authority"
                )
            if isinstance(generation, Mapping) and _generation_alive(generation):
                observed = _observe_generation(
                    generation,
                    spec=self.spec,
                    barrier_record=raw,
                )
                if observed is None:
                    raise TasteGINEControllerError(
                        "release-authorized generation disappeared during reopen"
                    )
                if (
                    sys.platform.startswith("linux")
                    and observed.get("last_observed_phase") != "startup_launcher"
                ):
                    self._write_state(
                        "RUNNING",
                        attempt=attempt,
                        launch_index=int(state.get("launch_index", 0)),
                        barrier_record=state.get("barrier_record"),
                        worker_generation=observed,
                        worker_log=state.get("worker_log"),
                        worker_log_binding=state.get("worker_log_binding"),
                        retries_used=attempt,
                        adopted_after_controller_restart=True,
                    )
                    return self._monitor(self._state_or_error(), process=None)
            # RELEASE_AUTHORIZED is deliberately still a pre-release phase.  If
            # the controller died before sending the one-time token, re-arm the
            # same launch/attempt and do not spend the science-loss retry.
            launch_index = int(state.get("launch_index", 0))
            record_path = self._barrier_paths(launch_index)[1]
            record = validate_startup_barrier_record(
                record_path, expected_target_argv=self.spec.worker_argv
            )
            if record.to_dict() != dict(raw):
                raise TasteGINEControllerError(
                    "release-authorized barrier record changed"
                )
            record = validate_reopenable_unreleased_barrier(
                record_path,
                expected_target_argv=self.spec.worker_argv,
                timeout_seconds=10.0,
            )
            process = self._launch(
                attempt,
                launch_index=launch_index,
                rearm=record,
                arming_already_durable=True,
            )
            return self._monitor(self._state_or_error(), process=process)
        if phase in {"RELEASED", "RUNNING", "RUNNING_TRAINER_ADOPTED"}:
            generation = state.get("worker_generation")
            raw = state.get("barrier_record")
            if (
                isinstance(generation, Mapping)
                and isinstance(raw, Mapping)
                and _generation_alive(generation)
            ):
                try:
                    observed = _observe_generation(
                        generation,
                        spec=self.spec,
                        barrier_record=raw,
                    )
                except (
                    TasteGINEControllerError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                ) as exc:
                    return self._supervise_identity_drift(
                        state, process=None, reason=str(exc)
                    )
                if observed is None:
                    return self._monitor(state, process=None)
                self._write_state(
                    "RUNNING",
                    attempt=attempt,
                    launch_index=int(state.get("launch_index", 0)),
                    barrier_record=raw,
                    worker_generation=observed,
                    worker_log=state.get("worker_log"),
                    worker_log_binding=state.get("worker_log_binding"),
                    retries_used=attempt,
                    adopted_after_controller_restart=True,
                )
                return self._monitor(self._state_or_error(), process=None)
            return self._monitor(state, process=None)
        if phase == "SUPERVISING_IDENTITY_DRIFT":
            return self._supervise_identity_drift(
                state,
                process=None,
                reason=str(state.get("reason", "worker identity drift")),
            )
        if phase == "SUPERVISING_TRAINER_IDENTITY_DRIFT":
            return self._supervise_trainer_identity_drift(
                state,
                process=None,
                reason=str(state.get("reason", "trainer identity drift")),
            )
        if phase in {"RETRY_PENDING", "WAITING_RESOURCES"}:
            if phase == "WAITING_RESOURCES" and state.get(
                "resource_generation_cap"
            ) is True:
                while not self._resource_deadline_expired():
                    time.sleep(self.spec.poll_seconds)
                self._write_state(
                    "FAILED",
                    attempt=attempt,
                    launch_index=int(state.get("launch_index", 0)),
                    retries_used=attempt,
                    reason="RESOURCE_WAIT_GLOBAL_DEADLINE_EXCEEDED",
                    returncode=75,
                )
                return 75
            if attempt + 1 >= self.spec.max_attempts:
                if phase == "RETRY_PENDING":
                    raise TasteGINEControllerError("retry-pending state exceeds attempt budget")
            next_attempt = attempt + 1 if phase == "RETRY_PENDING" else attempt
            process = self._launch(
                next_attempt,
                launch_index=int(state.get("launch_index", 0)),
            )
            return self._monitor(self._state_or_error(), process=process)
        if phase == "PUBLISHED_OUTPUT_ADOPTION_PENDING":
            validate_tastemolnet_published_output_adoption_readonly(
                self.root / PUBLISHED_ADOPTION_NAME,
                expected_output_dir=self.spec.output_dir,
                expected_training_state_root=self.spec.training_state_root,
            )
            process = self._launch(
                attempt,
                launch_index=int(state.get("launch_index", 0)),
            )
            return self._monitor(self._state_or_error(), process=process)
        raise TasteGINEControllerError(f"unsupported controller phase: {phase}")


def _spec_from_payload(payload: Mapping[str, Any]) -> TasteGINEControllerSpec:
    return TasteGINEControllerSpec(
        cid=str(payload["cid"]),
        controller_root=Path(str(payload["controller_root"])),
        project_root=Path(str(payload["project_root"])),
        output_dir=Path(str(payload["output_dir"])),
        training_state_root=Path(str(payload["training_state_root"])),
        worker_argv=tuple(str(value) for value in payload["worker_argv"]),
        source_identity=dict(payload["source_identity"]),
        environment_authority=dict(payload["environment_authority"]),
        config_files=tuple(dict(row) for row in payload["config_files"]),
        max_attempts=int(payload["max_attempts"]),
        poll_seconds=float(payload["poll_seconds"]),
        terminal_stability_seconds=float(payload["terminal_stability_seconds"]),
        resource_wait_deadline_seconds=int(
            payload.get("resource_wait_deadline_seconds", 604800)
        ),
    )


def validate_tastemolnet_gine_pass_readonly(
    root: str | Path,
    *,
    held_controller: TasteGINEPersistentController | None = None,
) -> dict[str, Any]:
    """Strictly reopen state/terminal/PASS and all terminal sources without writes."""

    controller_root = _absolute(root)
    root_fd: int | None = None
    lock_fd: int | None = None
    try:
        assert_no_symlink_components(
            controller_root, label="Taste controller terminal root"
        )
        root_before = os.lstat(controller_root)
        if not stat.S_ISDIR(root_before.st_mode) or stat.S_ISLNK(root_before.st_mode):
            raise TasteGINEControllerError(
                "terminal controller root is not one physical directory"
            )
        if held_controller is None:
            root_fd = os.open(
                controller_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            lock_fd = os.open(
                controller_root / ROOT_LOCK,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TasteGINEControllerError(
                    "terminal controller still has an active writer"
                ) from exc
        else:
            if held_controller.root != controller_root:
                raise TasteGINEControllerError(
                    "held controller root differs from terminal validator root"
                )
            held_controller.verify_authority()

        terminal_names = sorted(
            path.name
            for path in controller_root.iterdir()
            if "terminal" in path.name.casefold()
        )
        if terminal_names != [TERMINAL_NAME]:
            raise TasteGINEControllerError(
                "strict PASS closure has missing/extra terminal-named artifacts"
            )

        claim = _load_json(
            controller_root / ROOT_CLAIM, label="controller root claim"
        )
        spec_payload = _load_json(
            controller_root / SPEC_NAME, label="controller spec"
        )
        state = _load_json(
            controller_root / STATE_NAME, label="controller final state"
        )
        terminal_path = controller_root / TERMINAL_NAME
        terminal = _load_json(terminal_path, label="controller terminal")
        marker_text, marker_identity = _load_text_bound(
            controller_root / PASS_NAME, label="controller PASS marker"
        )
        spec_sha = stable_sha256(spec_payload)
        deadline_path = controller_root / RESOURCE_DEADLINE_NAME
        deadline = _load_json(deadline_path, label="resource-wait deadline")
        sentinel = controller_root / ROOT_SENTINEL
        lock = controller_root / ROOT_LOCK
        if (
            claim.get("schema_version") != CLAIM_SCHEMA
            or claim.get("root") != str(controller_root)
            or claim.get("root_identity") != _directory_identity(root_before)
            or claim.get("spec_sha256") != spec_sha
            or spec_payload.get("schema_version") != SCHEMA
            or spec_payload.get("environment_authority_sha256")
            != stable_sha256(spec_payload.get("environment_authority", {}))
            or state.get("schema_version") != STATE_SCHEMA
            or state.get("phase") != "PASS"
            or state.get("cid") != claim.get("cid")
            or state.get("cid") != spec_payload.get("cid")
            or state.get("spec_sha256") != spec_sha
            or state.get("root_claim_sha256")
            != sha256_file(controller_root / ROOT_CLAIM)
            or state.get("resource_deadline_sha256") != sha256_file(deadline_path)
            or state.get("resource_deadline_epoch_seconds")
            != deadline.get("deadline_epoch_seconds")
            or not _strict_int(state.get("attempt"))
            or not _strict_int(state.get("launch_index"))
            or not _strict_int(state.get("retries_used"))
            or state.get("attempt") < 0
            or state.get("launch_index") < 0
            or state.get("launch_index") >= MAX_STARTUP_GENERATIONS
            or state.get("retries_used") != state.get("attempt")
            or not isinstance(state.get("updated_at"), str)
            or not state.get("updated_at")
            or terminal.get("schema_version") != TERMINAL_SCHEMA
            or terminal.get("status") != "PASS"
            or terminal.get("cid") != spec_payload.get("cid")
            or terminal.get("spec_sha256") != spec_sha
            or not _strict_int(terminal.get("attempt"))
            or not _strict_int(terminal.get("launch_index"))
            or not _strict_int(terminal.get("process_loss_retries_used"))
            or terminal.get("attempt") != state.get("attempt")
            or terminal.get("launch_index") != state.get("launch_index")
            or terminal.get("process_loss_retries_used")
            != state.get("retries_used")
            or terminal.get("same_training_state_root")
            != spec_payload.get("training_state_root")
            or state.get("terminal_sha256") != sha256_file(terminal_path)
            or marker_text != "TASTEMOLNET_GINE_CONTROLLER_PASS\n"
            or _file_identity(os.lstat(sentinel))
            != claim.get("sentinel", {}).get("identity")
            or sha256_file(sentinel) != claim.get("sentinel", {}).get("sha256")
            or _file_identity(os.lstat(lock))
            != claim.get("lock", {}).get("identity")
            or deadline.get("schema_version") != RESOURCE_DEADLINE_SCHEMA
            or deadline.get("cid") != spec_payload.get("cid")
            or deadline.get("spec_sha256") != spec_sha
            or not _strict_int(deadline.get("duration_seconds"))
            or not _strict_int(deadline.get("started_epoch_seconds"))
            or not _strict_int(deadline.get("deadline_epoch_seconds"))
            or deadline.get("duration_seconds") <= 0
            or deadline.get("started_epoch_seconds") <= 0
            or deadline.get("deadline_epoch_seconds")
            - deadline.get("started_epoch_seconds")
            != deadline.get("duration_seconds")
        ):
            raise TasteGINEControllerError(
                "strict read-only PASS closure changed"
            )
        evidence = terminal.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TasteGINEControllerError("controller terminal evidence is untyped")
        spec = _spec_from_payload(spec_payload)
        if spec.payload() != spec_payload:
            raise TasteGINEControllerError("controller terminal spec is not canonical")
        if state["attempt"] >= spec.max_attempts:
            raise TasteGINEControllerError(
                "controller terminal attempt exceeds its frozen retry budget"
            )
        validator = held_controller or TasteGINEPersistentController(spec, resume=True)
        validator._validate_spec_sources()
        deep_evidence = validator._terminal_evidence()
        if deep_evidence != dict(evidence):
            raise TasteGINEControllerError(
                "controller terminal sources failed strict read-only reopen"
            )
        # Reopen every controller source after the external output/state scan.
        if (
            _load_json(controller_root / ROOT_CLAIM, label="controller root claim")
            != claim
            or _load_json(controller_root / SPEC_NAME, label="controller spec")
            != spec_payload
            or _load_json(controller_root / STATE_NAME, label="controller final state")
            != state
            or _load_json(terminal_path, label="controller terminal") != terminal
            or _load_json(deadline_path, label="resource-wait deadline") != deadline
        ):
            raise TasteGINEControllerError(
                "controller terminal closure changed during read-only reopen"
            )
        marker_after, marker_identity_after = _load_text_bound(
            controller_root / PASS_NAME, label="controller PASS marker"
        )
        root_after = os.lstat(controller_root)
        if (
            marker_after != marker_text
            or marker_identity_after != marker_identity
            or _directory_identity(root_after) != _directory_identity(root_before)
        ):
            raise TasteGINEControllerError(
                "controller PASS/root changed during read-only reopen"
            )
        if held_controller is not None:
            held_controller.verify_authority()
        return {
            "claim": claim,
            "spec": spec_payload,
            "state": state,
            "terminal": terminal,
            "marker_identity": marker_identity,
        }
    except (MolecularGNNResumeError, OSError, KeyError, TypeError, ValueError) as exc:
        raise TasteGINEControllerError(str(exc)) from exc
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)


def run_tastemolnet_gine_controller(
    spec: TasteGINEControllerSpec, *, resume: bool
) -> int:
    with TasteGINEPersistentController(spec, resume=resume) as controller:
        return controller.run()


def inspect_tastemolnet_gine_controller(root: str | Path) -> dict[str, Any]:
    """Typed, read-only status reopen; terminal PASS is deeply revalidated."""

    controller_root = _absolute(root)
    try:
        assert_no_symlink_components(
            controller_root, label="Taste controller status root"
        )
    except MolecularGNNResumeError as exc:
        raise TasteGINEControllerError(str(exc)) from exc
    root_before = os.lstat(controller_root)
    if not stat.S_ISDIR(root_before.st_mode) or stat.S_ISLNK(root_before.st_mode):
        raise TasteGINEControllerError("status root is not one physical directory")
    claim = _load_json(controller_root / ROOT_CLAIM, label="controller root claim")
    spec_payload = _load_json(controller_root / SPEC_NAME, label="controller spec")
    state = _load_json(controller_root / STATE_NAME, label="controller state")
    deadline_path = controller_root / RESOURCE_DEADLINE_NAME
    deadline = _load_json(deadline_path, label="resource-wait deadline")
    if (
        claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("root") != str(controller_root)
        or claim.get("root_identity") != _directory_identity(root_before)
        or claim.get("spec_sha256") != stable_sha256(spec_payload)
        or spec_payload.get("schema_version") != SCHEMA
        or state.get("schema_version") != STATE_SCHEMA
        or state.get("cid") != claim.get("cid")
        or state.get("cid") != spec_payload.get("cid")
        or state.get("spec_sha256") != stable_sha256(spec_payload)
        or state.get("root_claim_sha256")
        != sha256_file(controller_root / ROOT_CLAIM)
        or state.get("resource_deadline_sha256") != sha256_file(deadline_path)
        or state.get("resource_deadline_epoch_seconds")
        != deadline.get("deadline_epoch_seconds")
        or spec_payload.get("environment_authority_sha256")
        != stable_sha256(spec_payload.get("environment_authority", {}))
    ):
        raise TasteGINEControllerError("typed controller status closure changed")
    sentinel = controller_root / ROOT_SENTINEL
    lock = controller_root / ROOT_LOCK
    if (
        _file_identity(os.lstat(sentinel)) != claim.get("sentinel", {}).get("identity")
        or sha256_file(sentinel) != claim.get("sentinel", {}).get("sha256")
        or _file_identity(os.lstat(lock)) != claim.get("lock", {}).get("identity")
    ):
        raise TasteGINEControllerError("typed controller status root authority changed")
    root_after = os.lstat(controller_root)
    if (
        _directory_identity(root_before) != _directory_identity(root_after)
        or _load_json(controller_root / STATE_NAME, label="controller state") != state
    ):
        raise TasteGINEControllerError("controller root changed during status read")
    pass_path = controller_root / PASS_NAME
    if _terminal_signal_present(controller_root, state):
        validate_tastemolnet_gine_pass_readonly(controller_root)
    return {
        "schema_version": SCHEMA,
        "controller_root": str(controller_root),
        "cid": spec_payload["cid"],
        "phase": state["phase"],
        "attempt": state.get("attempt"),
        "launch_index": state.get("launch_index"),
        "pass": pass_path.is_file(),
        "typed_reopen": True,
    }


__all__ = [
    "CID_PATTERN",
    "REQUIRED_OUTPUT_FILES",
    "TasteGINEControllerError",
    "TasteGINEControllerSpec",
    "TasteGINEPersistentController",
    "inspect_tastemolnet_gine_controller",
    "run_tastemolnet_gine_controller",
    "validate_tastemolnet_gine_pass_readonly",
]
