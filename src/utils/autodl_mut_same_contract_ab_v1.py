"""Immutable one-shot spec for the bounded Mut same-contract trace A/B."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from uuid import UUID

from .autodl_mut_first_divergence_v1 import file_sha256, stable_sha256


SCHEMA = "mut_same_contract_trace_ab_task_spec_v1"
SOURCE_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
INSTRUMENTATION_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
STEPS = 500
POST_RELOAD_STEPS = 10
CANDIDATE_CAPACITY = 100_000
_SHA = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class MutSameContractABSpecError(RuntimeError):
    """The bounded A/B spec is stale, ambiguous, or unsafe."""


def _absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise MutSameContractABSpecError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise MutSameContractABSpecError(f"{field} must be normalized and absolute")
    return path


def _git_head(path: Path) -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MutSameContractABSpecError(f"cannot read Git HEAD: {path}") from exc


def _git_status(path: Path) -> str:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MutSameContractABSpecError(f"cannot read Git status: {path}") from exc


def _tree_sha(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: file_sha256(path / name) for name in names}


def build_same_contract_ab_spec(
    template: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    value = dict(template)
    required = {
        "task_id",
        "attempt_uuid",
        "controller_project_root",
        "controller_commit",
        "python",
        "runner_path",
        "legacy_project_root",
        "execution_project_root",
        "historical_artifact_root",
        "upstream_root",
        "dataset_dir",
        "gnn_checkpoint",
        "distance_checkpoint",
        "rf_oracle",
        "run_root",
        "output_dir",
        "control_root",
        "lease_path",
        "gpu_lock_root",
        "gpu_uuid",
        "gpu_index",
    }
    if set(value) != required:
        raise MutSameContractABSpecError(
            f"A/B template keys differ: missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    try:
        attempt = UUID(str(value["attempt_uuid"]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise MutSameContractABSpecError("attempt_uuid must be UUIDv4") from exc
    if attempt.version != 4 or str(attempt) != value["attempt_uuid"]:
        raise MutSameContractABSpecError("attempt_uuid must be canonical UUIDv4")
    if _GIT_SHA.fullmatch(str(value["controller_commit"])) is None:
        raise MutSameContractABSpecError("controller_commit must be full Git SHA")
    paths = {
        field: _absolute(value[field], field=field)
        for field in required
        if field
        not in {
            "task_id",
            "attempt_uuid",
            "controller_commit",
            "gpu_index",
            "gpu_uuid",
        }
    }
    if value["gpu_index"] != 0 or isinstance(value["gpu_index"], bool):
        raise MutSameContractABSpecError("Mut A/B is pinned to physical GPU0")
    if _GPU_UUID.fullmatch(str(value["gpu_uuid"])) is None:
        raise MutSameContractABSpecError("gpu_uuid must be one physical GPU UUID")
    if paths["lease_path"] == paths["gpu_lock_root"] / f"gpu-{value['gpu_uuid']}.lock":
        raise MutSameContractABSpecError("owner lease and global GPU UUID lock must differ")
    if check_files:
        expected_runner = (
            paths["controller_project_root"]
            / "scripts/autodl/run_mut_trace_mode_equivalence.py"
        )
        if paths["runner_path"] != expected_runner:
            raise MutSameContractABSpecError("A/B runner escaped controller worktree")
        if _git_head(paths["controller_project_root"]) != value["controller_commit"]:
            raise MutSameContractABSpecError("controller worktree commit changed")
        if _git_head(paths["legacy_project_root"]) != SOURCE_COMMIT:
            raise MutSameContractABSpecError("legacy review worktree commit changed")
        if _git_head(paths["execution_project_root"]) != INSTRUMENTATION_COMMIT:
            raise MutSameContractABSpecError("A/B science worktree commit changed")
        if _git_head(paths["upstream_root"]) != UPSTREAM_COMMIT:
            raise MutSameContractABSpecError("upstream COMRECGC commit changed")
        for field in (
            "controller_project_root",
            "legacy_project_root",
            "execution_project_root",
            "upstream_root",
        ):
            if _git_status(paths[field]):
                raise MutSameContractABSpecError(
                    f"A/B Git input has dirty or shadow source: {field}"
                )
        for field in (
            "python",
            "runner_path",
            "gnn_checkpoint",
            "distance_checkpoint",
            "rf_oracle",
        ):
            if not paths[field].is_file():
                raise MutSameContractABSpecError(f"A/B input is absent: {field}")
        for field in (
            "legacy_project_root",
            "execution_project_root",
            "historical_artifact_root",
            "upstream_root",
            "dataset_dir",
        ):
            if not paths[field].is_dir() or paths[field].is_symlink():
                raise MutSameContractABSpecError(f"A/B directory is absent/indirect: {field}")
        for field in ("run_root", "output_dir", "control_root"):
            if paths[field].exists() or paths[field].is_symlink():
                raise MutSameContractABSpecError(f"A/B output must be fresh: {field}")
        bound_hashes = {
            "runner": file_sha256(paths["runner_path"]),
            "historical_manifest": file_sha256(
                paths["historical_artifact_root"] / "run_manifest.json"
            ),
            "gnn_checkpoint": file_sha256(paths["gnn_checkpoint"]),
            "distance_checkpoint": file_sha256(paths["distance_checkpoint"]),
            "rf_oracle": file_sha256(paths["rf_oracle"]),
            **{
                f"dataset/{key}": digest
                for key, digest in _tree_sha(
                    paths["dataset_dir"],
                    ("dataset_summary.json", "generation_source_graphs.pt"),
                ).items()
            },
        }
    else:
        bound_hashes = {}
    spec: dict[str, Any] = {
        "schema_version": SCHEMA,
        **value,
        "source_algorithm_commit": SOURCE_COMMIT,
        "instrumentation_commit": INSTRUMENTATION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "steps": STEPS,
        "post_reload_steps": POST_RELOAD_STEPS,
        "candidate_capacity": CANDIDATE_CAPACITY,
        "trace_modes": ["on", "off"],
        "arms_sequential": True,
        "resume_parity_separate": True,
        "fresh_50k_started": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "required_environment": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RUN_LLM_ABLATION": "0",
            "RUN_GNN_ABLATION": "0",
        },
        "bound_file_sha256s": bound_hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    spec["spec_sha256"] = stable_sha256(spec)
    return validate_same_contract_ab_spec(spec, check_files=check_files)


def validate_same_contract_ab_spec(
    raw: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != SCHEMA:
        raise MutSameContractABSpecError("A/B spec schema changed")
    observed = value.get("spec_sha256")
    expected = stable_sha256(
        {key: item for key, item in value.items() if key != "spec_sha256"}
    )
    if observed != expected or not isinstance(observed, str) or _SHA.fullmatch(observed) is None:
        raise MutSameContractABSpecError("A/B spec self hash changed")
    frozen = {
        "source_algorithm_commit": SOURCE_COMMIT,
        "instrumentation_commit": INSTRUMENTATION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "steps": 500,
        "post_reload_steps": 10,
        "candidate_capacity": 100_000,
        "trace_modes": ["on", "off"],
        "arms_sequential": True,
        "resume_parity_separate": True,
        "fresh_50k_started": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    changed = [key for key, expected_value in frozen.items() if value.get(key) != expected_value]
    if changed:
        raise MutSameContractABSpecError(f"A/B frozen contract changed: {changed}")
    environment = value.get("required_environment")
    if not isinstance(environment, Mapping) or environment.get("PYTHONHASHSEED") != "0":
        raise MutSameContractABSpecError("A/B deterministic environment changed")
    path_fields = (
        "controller_project_root",
        "python",
        "runner_path",
        "legacy_project_root",
        "execution_project_root",
        "historical_artifact_root",
        "upstream_root",
        "dataset_dir",
        "gnn_checkpoint",
        "distance_checkpoint",
        "rf_oracle",
        "run_root",
        "output_dir",
        "control_root",
        "lease_path",
        "gpu_lock_root",
    )
    paths = {field: _absolute(value.get(field), field=field) for field in path_fields}
    if value.get("gpu_index") != 0 or isinstance(value.get("gpu_index"), bool):
        raise MutSameContractABSpecError("Mut A/B is pinned to physical GPU0")
    if _GPU_UUID.fullmatch(str(value.get("gpu_uuid") or "")) is None:
        raise MutSameContractABSpecError("A/B physical GPU UUID changed")
    if paths["lease_path"] == paths["gpu_lock_root"] / f"gpu-{value['gpu_uuid']}.lock":
        raise MutSameContractABSpecError("owner lease and global GPU UUID lock must differ")
    if check_files:
        expected_runner = (
            paths["controller_project_root"]
            / "scripts/autodl/run_mut_trace_mode_equivalence.py"
        )
        if paths["runner_path"] != expected_runner:
            raise MutSameContractABSpecError("A/B runner escaped controller worktree")
        expected_hashes = value.get("bound_file_sha256s")
        if not isinstance(expected_hashes, Mapping):
            raise MutSameContractABSpecError("A/B file bindings are absent")
        observed_hashes = {
            "runner": file_sha256(paths["runner_path"]),
            "historical_manifest": file_sha256(
                paths["historical_artifact_root"] / "run_manifest.json"
            ),
            "gnn_checkpoint": file_sha256(paths["gnn_checkpoint"]),
            "distance_checkpoint": file_sha256(paths["distance_checkpoint"]),
            "rf_oracle": file_sha256(paths["rf_oracle"]),
            **{
                f"dataset/{key}": digest
                for key, digest in _tree_sha(
                    paths["dataset_dir"],
                    ("dataset_summary.json", "generation_source_graphs.pt"),
                ).items()
            },
        }
        if dict(expected_hashes) != observed_hashes:
            raise MutSameContractABSpecError("A/B bound input bytes changed")
        if _git_head(paths["controller_project_root"]) != value["controller_commit"]:
            raise MutSameContractABSpecError("controller worktree commit changed")
        if _git_head(paths["legacy_project_root"]) != SOURCE_COMMIT:
            raise MutSameContractABSpecError("legacy worktree commit changed")
        if _git_head(paths["execution_project_root"]) != INSTRUMENTATION_COMMIT:
            raise MutSameContractABSpecError("science worktree commit changed")
        if _git_head(paths["upstream_root"]) != UPSTREAM_COMMIT:
            raise MutSameContractABSpecError("upstream worktree commit changed")
        for field in (
            "controller_project_root",
            "legacy_project_root",
            "execution_project_root",
            "upstream_root",
        ):
            if _git_status(paths[field]):
                raise MutSameContractABSpecError(
                    f"A/B Git input has dirty or shadow source: {field}"
                )
    return value


def same_contract_ab_command(spec: Mapping[str, Any]) -> list[str]:
    value = validate_same_contract_ab_spec(spec, check_files=False)
    return [
        str(value["python"]),
        "-I",
        "-B",
        str(value["runner_path"]),
        "--config",
        str(Path(value["controller_project_root"]) / "configs/hpc.yaml"),
        "--set",
        "inference.fallback_to_heuristic=false",
        "run-pair",
        "--python",
        str(value["python"]),
        "--legacy-project-root",
        str(value["legacy_project_root"]),
        "--execution-project-root",
        str(value["execution_project_root"]),
        "--run-root",
        str(value["run_root"]),
        "--output-dir",
        str(value["output_dir"]),
        "--historical-artifact-root",
        str(value["historical_artifact_root"]),
        "--rf-oracle",
        str(value["rf_oracle"]),
        "--upstream-root",
        str(value["upstream_root"]),
        "--dataset-dir",
        str(value["dataset_dir"]),
        "--gnn-checkpoint",
        str(value["gnn_checkpoint"]),
        "--distance-checkpoint",
        str(value["distance_checkpoint"]),
        "--parent-limit",
        "1448",
        "--device",
        "cuda:0",
        "--batch-size",
        "128",
    ]


__all__ = [
    "MutSameContractABSpecError",
    "SCHEMA",
    "build_same_contract_ab_spec",
    "same_contract_ab_command",
    "validate_same_contract_ab_spec",
]
