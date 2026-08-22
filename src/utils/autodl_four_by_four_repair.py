"""Build the bounded four-by-four AutoDL repair continuation.

The repair controller is intentionally smaller than the main campaign.  It
contains only the four failed terminal closures that can be retried without
changing a completed scientific result:

* the native BACE ComRecGC GINE route and its artifact-only standardizer;
* Mutagenicity GCF calibration, held-out evaluation, and standardizer from an
  exact previously frozen candidate package;
* fresh AIDS/Mutagenicity ComRecGC threshold verification and standardized
  continuation from immutable recovered generations; and
* an artifact-only retry of the already-passing BACE Ours B14 freeze.

Every adopted source is checked both while the manifest is built and again as
the first controller task.  A source task must be PASS, its actual passing
attempt must equal the explicit absolute root, and Linux procfs must expose no
writable descriptor below that root.  The module never mutates an old
controller, attempt, PASS root, or failure root.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from scripts.autodl.build_bace_cell_standardization_tasks import (
    REQUIRED_FILES as BACE_STANDARDIZED_REQUIRED_FILES,
)
from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerError,
    load_controller_manifest,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (
    build_bace_baseline_generic_controller_fragment,
)
from src.eval.am_legacy_standardization import scan_live_writers


SPEC_SCHEMA = "four_by_four_repair_spec_v1"
MANIFEST_CONTROLLER_ID = "four_methods_four_datasets_repair_v1"
SOURCE_CONTROLLER_TERMINALS = {
    "bace_b14": "bace_b14_frozen",
    "mut_gcf_freeze": "mut_gcf_legacy_freeze",
}
ARTIFACT_TERMINALS = {
    "mut_comrec_generation": "mutagenicity",
    "aids_comrec_generation": "aids",
}
SOURCE_TASK_IDS = {
    "bace_b14": "repair_source_bace_b14",
    "mut_gcf_freeze": "repair_source_mut_gcf_freeze",
    "mut_comrec_generation": "repair_source_mut_comrec_generation",
    "aids_comrec_generation": "repair_source_aids_comrec_generation",
}
BACE_STANDARDIZED_FILES = [*BACE_STANDARDIZED_REQUIRED_FILES]
HEX40 = re.compile(r"[0-9a-f]{40}")
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")


class RepairManifestError(ValueError):
    """A repair manifest would weaken source or execution invariants."""


@dataclass(frozen=True)
class SourceEvidence:
    name: str
    kind: str
    output_root: Path
    audit: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"JSON source is not a physical nonempty file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"Invalid JSON object: {source}") from exc
    if not isinstance(payload, dict):
        raise RepairManifestError(f"Expected one JSON object: {source}")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairManifestError(f"{label} must be one object")
    return value


def _absolute_path(value: Any, *, label: str, kind: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RepairManifestError(f"{label} must be absolute: {path}")
    if kind == "fresh":
        return path.resolve(strict=False)
    if path.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if kind == "file" and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise RepairManifestError(f"{label} must be a nonempty file: {resolved}")
    if kind == "dir" and not resolved.is_dir():
        raise RepairManifestError(f"{label} must be a directory: {resolved}")
    return resolved


def _nested(spec: Mapping[str, Any], *parts: str) -> Any:
    value: Any = spec
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        if not isinstance(value, Mapping) or part not in value:
            raise RepairManifestError(f"Missing repair spec field: {'.'.join(traversed)}")
        value = value[part]
    return value


def _required_files(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RepairManifestError(f"Unsafe required source file: {name}")
        logical = root / relative
        if logical.is_symlink():
            raise RepairManifestError(f"Required source file may not be a symlink: {logical}")
        source = logical.resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RepairManifestError(f"Required source file escapes root: {logical}") from exc
        if not source.is_file() or source.stat().st_size <= 0:
            raise RepairManifestError(f"Required source file is empty: {source}")
        result[name] = {
            "path": str(source),
            "size": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    return result


def _assert_physical_root(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute():
        raise RepairManifestError(f"{label} must be absolute: {logical}")
    if logical.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink: {logical}")
    root = logical.resolve(strict=True)
    if not root.is_dir():
        raise RepairManifestError(f"{label} must be a directory: {root}")
    return root


def verify_controller_terminal(
    *,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    task_id: str,
    expected_output_root: str | Path,
    required_files: Sequence[str],
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Verify one exact PASS task and its immutable passing attempt."""

    if SAFE_ID.fullmatch(task_id) is None:
        raise RepairManifestError(f"Unsafe source task id: {task_id!r}")
    manifest_logical = Path(source_manifest).expanduser()
    if not manifest_logical.is_absolute() or manifest_logical.is_symlink():
        raise RepairManifestError(
            f"Source controller manifest must be an absolute physical file: {manifest_logical}"
        )
    manifest_path = manifest_logical.resolve(strict=True)
    manifest = load_controller_manifest(manifest_path)
    root = _assert_physical_root(source_controller_root, label="source_controller_root")
    if manifest_path.parent.name != "manifests":
        raise RepairManifestError(
            "Source manifest must use <namespace>/manifests/<manifest>"
        )
    namespace_root = manifest_path.parent.parent
    if namespace_root.is_symlink() or manifest_path.parent.is_symlink():
        raise RepairManifestError("Source controller namespace must be physical")
    expected_controller_root = (namespace_root / manifest.controller_id).resolve(
        strict=True
    )
    if root != expected_controller_root:
        raise RepairManifestError(
            "Source controller root does not match the source manifest namespace"
        )
    snapshot_path = root / "controller_manifest.json"
    snapshot = _read_object(snapshot_path)
    if snapshot.get("controller_id") != manifest.controller_id:
        raise RepairManifestError("Source controller snapshot controller_id mismatch")
    recorded_manifest = snapshot.get("source_manifest")
    if not isinstance(recorded_manifest, str):
        raise RepairManifestError("Source controller snapshot has no source_manifest")
    if Path(recorded_manifest).expanduser().resolve(strict=True) != manifest_path:
        raise RepairManifestError("Source controller snapshot points to another manifest")
    if snapshot.get("source_manifest_sha256") != manifest.sha256:
        raise RepairManifestError("Source controller snapshot manifest SHA256 mismatch")
    if task_id not in manifest.by_id:
        raise RepairManifestError(f"Source controller has no task {task_id}")

    task_root = root / "tasks" / task_id
    state_path = task_root / "state.json"
    gate_path = task_root / "gate.json"
    task_manifest_path = task_root / "manifest.json"
    state = _read_object(state_path)
    gate = _read_object(gate_path)
    task_manifest = _read_object(task_manifest_path)
    if state.get("state") != "PASS":
        raise RepairManifestError(
            f"Source terminal {task_id} is not PASS: {state.get('state')!r}"
        )
    if gate.get("status") != "PASS":
        raise RepairManifestError(
            f"Source terminal gate {task_id} is not PASS: {gate.get('status')!r}"
        )
    if task_manifest.get("controller_manifest_sha256") != manifest.sha256:
        raise RepairManifestError("Source task manifest belongs to another controller manifest")
    instances = state.get("instances")
    if not isinstance(instances, Mapping) or len(instances) != 1:
        raise RepairManifestError("Source terminal must expose one unambiguous passing instance")
    instance = next(iter(instances.values()))
    if not isinstance(instance, Mapping) or instance.get("state") != "PASS":
        raise RepairManifestError("Source terminal passing instance is not PASS")
    recorded_output = instance.get("expected_output")
    if not isinstance(recorded_output, str) or not recorded_output:
        raise RepairManifestError("Source terminal passing instance has no expected_output")
    expected = _assert_physical_root(expected_output_root, label="expected_output_root")
    if Path(recorded_output).expanduser().resolve(strict=True) != expected:
        raise RepairManifestError(
            "Source terminal output mismatch: "
            f"state={recorded_output!r} expected={str(expected)!r}"
        )
    files = _required_files(expected, required_files)
    try:
        writer_audit = scan_live_writers(expected, proc_root=proc_root)
    except Exception as exc:
        raise RepairManifestError(f"Source terminal writer audit failed: {exc}") from exc
    return {
        "schema_version": "four_by_four_repair_source_terminal_v1",
        "status": "PASS",
        "kind": "controller_terminal",
        "source_controller_id": manifest.controller_id,
        "source_controller_manifest": str(manifest_path),
        "source_controller_manifest_sha256": manifest.sha256,
        "source_controller_root": str(root),
        "source_task_id": task_id,
        "source_task_state_sha256": sha256_file(state_path),
        "source_task_gate_sha256": sha256_file(gate_path),
        "source_task_manifest_sha256": sha256_file(task_manifest_path),
        "source_output_root": str(expected),
        "required_files": files,
        "live_writer_audit": writer_audit,
        "verified_at": _utc_now(),
    }


def verify_comrecgc_generation_terminal(
    *,
    dataset: str,
    expected_output_root: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Verify the small PASS closure around one recovered COMRECGC payload.

    The potentially large ``counterfactuals.pt`` is deliberately not hashed
    here.  The scientific continuation already computes that payload hash
    exactly once while holding before/after stat snapshots.
    """

    if dataset not in {"aids", "mutagenicity"}:
        raise RepairManifestError(f"Unsupported COMRECGC source dataset: {dataset}")
    root = _assert_physical_root(expected_output_root, label="generation output root")
    names = (
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "freeze_only_recovery.json",
        "frozen_payload_closure_audit.json",
        "adoption_manifest.json",
        "PASS",
    )
    files = _required_files(root, names)
    run_manifest = _read_object(root / "run_manifest.json")
    complete = _read_object(root / "_RUN_COMPLETE.json")
    recovery = _read_object(root / "freeze_only_recovery.json")
    closure = _read_object(root / "frozen_payload_closure_audit.json")
    adoption = _read_object(root / "adoption_manifest.json")
    checks = {
        "run_manifest.dataset": run_manifest.get("dataset") == dataset,
        "run_manifest.mode": run_manifest.get("mode") == "full",
        "run_manifest.generation_mode": (
            run_manifest.get("generation_mode") == "adopted_read_only_cache"
        ),
        "_RUN_COMPLETE.run_complete": complete.get("run_complete") is True,
        "_RUN_COMPLETE.freeze_only_recovery": (
            complete.get("freeze_only_recovery") is True
        ),
        "freeze_only_recovery.recovery_completed": (
            recovery.get("recovery_completed") is True
        ),
        "freeze_only_recovery.algorithm_rerun": (
            recovery.get("algorithm_rerun") is False
        ),
        "frozen_payload_closure.closure_complete": (
            closure.get("closure_complete") is True
        ),
        "frozen_payload_closure.post_write_reload_verified": (
            closure.get("post_write_reload_verified") is True
        ),
        "adoption_manifest.generation_mode": (
            adoption.get("generation_mode") == "adopted_read_only_cache"
        ),
        "PASS": (root / "PASS").read_text(encoding="utf-8").strip() == "PASS",
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RepairManifestError(
            f"Recovered COMRECGC generation terminal is not PASS: {failed}"
        )
    try:
        writer_audit = scan_live_writers(root, proc_root=proc_root)
    except Exception as exc:
        raise RepairManifestError(f"Generation source writer audit failed: {exc}") from exc
    return {
        "schema_version": "four_by_four_repair_generation_terminal_v1",
        "status": "PASS",
        "kind": "artifact_terminal",
        "dataset": dataset,
        "source_output_root": str(root),
        "checks": checks,
        "required_files": files,
        "large_payload_sha256_computed": False,
        "large_payload_hash_policy": "scientific_continuation_computes_exactly_once",
        "live_writer_audit": writer_audit,
        "verified_at": _utc_now(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_source_adoption(
    *, name: str, evidence: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    if name not in SOURCE_TASK_IDS:
        raise RepairManifestError(f"Unknown repair source name: {name}")
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        raise RepairManifestError("Source adoption output must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Source adoption output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o755)
    payload = {
        "schema_version": "four_by_four_repair_source_adoption_v1",
        "status": "PASS",
        "source_name": name,
        "source_evidence": dict(evidence),
        "published_at": _utc_now(),
    }
    _atomic_json(destination / "source_adoption.json", payload)
    pass_path = destination / "PASS"
    descriptor = os.open(pass_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _git_head(project_root: Path) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if HEX40.fullmatch(value) is None:
        raise RepairManifestError(f"Project HEAD is not a full commit: {value!r}")
    return value


def _assert_required_commits(project_root: Path, commits: Any) -> list[str]:
    if commits is None:
        return []
    if not isinstance(commits, list) or any(
        not isinstance(value, str) or HEX40.fullmatch(value) is None
        for value in commits
    ):
        raise RepairManifestError("required_execution_commits must be full commit SHAs")
    for commit in commits:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RepairManifestError(
                f"Execution worktree does not contain required fix commit: {commit}"
            )
    return list(commits)


def _source_task(
    *,
    source: SourceEvidence,
    project_root: Path,
    fresh_root: Path,
    source_manifest: Path | None,
    source_controller_root: Path | None,
    task_id: str | None,
) -> dict[str, Any]:
    adoption_id = SOURCE_TASK_IDS[source.name]
    expected = str(fresh_root / "source-adoptions" / source.name / "attempt-{attempt}")
    script = "{project_root}/scripts/autodl/build_four_by_four_repair_manifest.py"
    command = ["{python}", script]
    if source.kind == "controller_terminal":
        assert source_manifest is not None and source_controller_root is not None
        assert task_id is not None
        required = list(source.audit["required_files"])
        command.extend(
            [
                "verify-controller-terminal",
                "--source-name",
                source.name,
                "--source-manifest",
                str(source_manifest),
                "--source-controller-root",
                str(source_controller_root),
                "--task-id",
                task_id,
                "--expected-output-root",
                str(source.output_root),
            ]
        )
        for name in required:
            command.extend(["--required-file", name])
        input_manifest = str(source_controller_root / "tasks" / task_id / "gate.json")
    else:
        command.extend(
            [
                "verify-generation-terminal",
                "--source-name",
                source.name,
                "--dataset",
                str(source.audit["dataset"]),
                "--expected-output-root",
                str(source.output_root),
            ]
        )
        input_manifest = str(source.output_root / "run_manifest.json")
    command.extend(["--output-dir", "{task_output}"])
    return {
        "id": adoption_id,
        "dataset": "repair-source-audit",
        "stage": "FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION",
        "runner_dataset": f"repair-source-{source.name}",
        "runner_stage": "FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION",
        "depends_on": [],
        "resource": "cpu",
        "priority": 1 + list(SOURCE_TASK_IDS).index(source.name),
        "data_splits": [],
        "manifest_only": True,
        "command": command,
        "input_manifest": input_manifest,
        "expected_output": expected,
        "required_output_files": ["source_adoption.json", "PASS"],
        "required_log_marker": (
            f"[FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION_PASS] source={source.name}"
        ),
        "environment": {
            "PYTHONPATH": str(project_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RUN_TASTEMOLNET": "0",
        },
    }


def _bace_ours_standardizer(
    *, spec: Mapping[str, Any], b14_root: Path, fresh_root: Path, checkpoint: Path
) -> dict[str, Any]:
    task_id = "bace_ours_standardized"
    dependency = SOURCE_TASK_IDS["bace_b14"]
    command = [
        "{python}",
        "{project_root}/scripts/autodl/standardize_bace_frozen_cell.py",
        "--config",
        "configs/hpc.yaml",
        "--method",
        "Ours",
        "--source-final-root",
        str(b14_root),
        "--gnn-checkpoint",
        str(checkpoint),
        "--output-dir",
        "{task_output}",
    ]
    expected_hashes = _mapping(
        _nested(spec, "bace", "expected_hashes"), label="bace.expected_hashes"
    )
    for field, flag in (
        ("dataset", "--expected-dataset-hash"),
        ("split", "--expected-split-hash"),
        ("molclr", "--expected-molclr-hash"),
        ("threshold", "--expected-threshold-hash"),
    ):
        value = expected_hashes.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise RepairManifestError(f"bace.expected_hashes.{field} is invalid")
            command.extend([flag, value])
    return {
        "id": task_id,
        "dataset": "bace",
        "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
        "runner_dataset": "paper-cell-bace-ours-repair",
        "runner_stage": "BACE_FROZEN_CELL_STANDARDIZATION",
        "depends_on": [dependency],
        "resource": "cpu",
        "priority": 20,
        "data_splits": [],
        "manifest_only": True,
        "command": command,
        "input_manifest": "{dep_" + dependency + "_output}/source_adoption.json",
        "config_files": [str(b14_root / "FINAL_PASS.json")],
        "expected_output": str(
            fresh_root / "cells/bace/ours/standardized/attempt-{attempt}"
        ),
        "required_output_files": [
            *BACE_STANDARDIZED_FILES,
            "table2_ours_k10.csv",
        ],
        "required_log_marker": "[BACE_FROZEN_CELL_STANDARDIZATION_PASS] method=Ours",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "RUN_TASTEMOLNET": "0",
        },
        "semantic_failure_markers": [
            "frozen contract differs",
            "identity sha256 changed",
            "raw test opened",
            "threshold_config_hash differs",
            "rf-free bace gine contract",
            "artifact-only replay",
        ],
    }


def _bace_comrecgc_tasks(
    *, spec: Mapping[str, Any], fresh_root: Path, project_root: Path, python: Path
) -> list[dict[str, Any]]:
    bace = _mapping(_nested(spec, "bace"), label="bace")
    checkpoint = _absolute_path(bace.get("gnn_checkpoint"), label="bace.gnn_checkpoint", kind="dir")
    dataset_dir = _absolute_path(bace.get("dataset_dir"), label="bace.dataset_dir", kind="dir")
    calibration = _absolute_path(
        bace.get("calibration_split"), label="bace.calibration_split", kind="file"
    )
    test = _absolute_path(bace.get("test_split"), label="bace.test_split", kind="file")
    molclr_root = _absolute_path(bace.get("molclr_root"), label="bace.molclr_root", kind="dir")
    molclr_checkpoint = _absolute_path(
        bace.get("molclr_checkpoint"), label="bace.molclr_checkpoint", kind="file"
    )
    neurosed = _absolute_path(
        bace.get("neurosed_checkpoint"), label="bace.neurosed_checkpoint", kind="file"
    )
    official = _absolute_path(
        bace.get("comrecgc_official_root"),
        label="bace.comrecgc_official_root",
        kind="dir",
    )
    model_card = checkpoint / "model_card.json"
    _absolute_path(model_card, label="bace GINE model_card", kind="file")
    fragment = build_bace_baseline_generic_controller_fragment(
        method="ComRecGC",
        python=python,
        project_root=project_root,
        output_root=fresh_root / "bace/comrecgc/native",
        gnn_checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        calibration_split=calibration,
        test_split=test,
        molclr_root=molclr_root,
        molclr_checkpoint=molclr_checkpoint,
        neurosed_checkpoint=neurosed,
        official_root=official,
        omp_threads=int(bace.get("omp_threads", 4)),
    )
    tasks = [dict(task) for task in fragment["tasks"]]
    standardizer_command = [
        "{python}",
        "{project_root}/scripts/autodl/standardize_bace_frozen_cell.py",
        "--config",
        "configs/hpc.yaml",
        "--method",
        "ComRecGC",
        "--source-final-root",
        "{dep_bace_comrecgc_final_freeze_output}",
        "--gnn-checkpoint",
        str(checkpoint),
        "--output-dir",
        "{task_output}",
    ]
    expected_hashes = _mapping(bace.get("expected_hashes"), label="bace.expected_hashes")
    for field, flag in (
        ("dataset", "--expected-dataset-hash"),
        ("split", "--expected-split-hash"),
        ("molclr", "--expected-molclr-hash"),
        ("threshold", "--expected-threshold-hash"),
    ):
        value = expected_hashes.get(field)
        if value is not None:
            standardizer_command.extend([flag, str(value)])
    tasks.append(
        {
            "id": "bace_comrecgc_standardized",
            "dataset": "bace",
            "stage": "BACE_FROZEN_CELL_STANDARDIZATION",
            "runner_dataset": "paper-cell-bace-comrecgc-repair",
            "runner_stage": "BACE_FROZEN_CELL_STANDARDIZATION",
            "depends_on": ["bace_comrecgc_final_freeze"],
            "resource": "cpu",
            "priority": 162,
            "data_splits": [],
            "manifest_only": True,
            "command": standardizer_command,
            "input_manifest": (
                "{dep_bace_comrecgc_final_freeze_output}/FINAL_PASS.json"
            ),
            "expected_output": str(
                fresh_root / "cells/bace/comrecgc/standardized/attempt-{attempt}"
            ),
            "required_output_files": [
                *BACE_STANDARDIZED_FILES,
                "table2_comrecgc_k10.csv",
            ],
            "required_log_marker": (
                "[BACE_FROZEN_CELL_STANDARDIZATION_PASS] method=ComRecGC"
            ),
            "environment": {
                "PYTHONPATH": "{project_root}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "RUN_TASTEMOLNET": "0",
            },
        }
    )
    return tasks


def _mut_gcf_tasks(
    *, spec: Mapping[str, Any], source_root: Path, fresh_root: Path
) -> list[dict[str, Any]]:
    values = _mapping(_nested(spec, "mut_gcf"), label="mut_gcf")
    calibration_csv = _absolute_path(
        values.get("calibration_csv"), label="mut_gcf.calibration_csv", kind="file"
    )
    test_csv = _absolute_path(values.get("test_csv"), label="mut_gcf.test_csv", kind="file")
    teacher = _absolute_path(values.get("teacher_path"), label="mut_gcf.teacher_path", kind="file")
    molclr_root = _absolute_path(values.get("molclr_root"), label="mut_gcf.molclr_root", kind="dir")
    molclr_checkpoint = _absolute_path(
        values.get("molclr_checkpoint"), label="mut_gcf.molclr_checkpoint", kind="file"
    )
    dependency = SOURCE_TASK_IDS["mut_gcf_freeze"]
    calibration_id = "mut_gcf_legacy_calibration"
    heldout_id = "mut_gcf_legacy_heldout"
    calibration_output = str(
        fresh_root / "mutagenicity/gcfexplainer/calibration/attempt-{attempt}"
    )
    heldout_output = str(
        fresh_root / "mutagenicity/gcfexplainer/heldout/attempt-{attempt}"
    )
    standardized_output = str(
        fresh_root / "cells/mutagenicity/gcfexplainer/standardized/attempt-{attempt}"
    )
    common = {
        "FROZEN_ROOT": str(source_root),
        "FULLGRAPH_CANDIDATES_PATH": str(source_root / "export/selected_top20.csv"),
        "FROZEN_MANIFEST": str(source_root / "frozen_candidate_manifest.json"),
        "TEACHER_PATH": str(teacher),
        "MOLCLR_ROOT": str(molclr_root),
        "MOLCLR_CKPT": str(molclr_checkpoint),
        "THRESHOLDS_JSON": str(source_root / "matched_thresholds.json"),
        "AUTODL_PYTHON": "{python}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUN_TASTEMOLNET": "0",
    }
    calibration = {
        "id": calibration_id,
        "dataset": "mutagenicity",
        "stage": "AM_MUT_GCF_CALIBRATION_FREEZE",
        "runner_dataset": "mutagenicity-rf-gcf-repair",
        "runner_stage": "AM_MUT_GCF_CALIBRATION_FREEZE",
        "depends_on": [dependency],
        "resource": "gpu",
        "priority": 30,
        "data_splits": ["calibration"],
        "freezes_selector": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_mut_gcf_legacy_evaluation.sh",
        ],
        "input_manifest": "{dep_" + dependency + "_output}/source_adoption.json",
        "config_files": [
            str(source_root / "frozen_candidate_manifest.json"),
            str(source_root / "matched_thresholds.json"),
        ],
        "expected_output": calibration_output,
        "required_output_files": ["audit.json", "run_manifest.json", "_RUN_COMPLETE.json"],
        "required_log_marker": "[MUT_GCF_LEGACY_CALIBRATION_PASS]",
        "environment": {
            **common,
            "ACTION": "calibration",
            "CALIBRATION_CSV": str(calibration_csv),
            "OUTPUT_ROOT": "{task_output}",
            "WNODE_CACHE_DB": "{task_output}/_cache/wnode.sqlite3",
            "NODE_EMB_CACHE_DIR": "{task_output}/_cache/node_embeddings",
        },
    }
    heldout = {
        "id": heldout_id,
        "dataset": "mutagenicity",
        "stage": "AM_MUT_GCF_HELDOUT_EVAL",
        "runner_dataset": "mutagenicity-rf-gcf-repair",
        "runner_stage": "AM_MUT_GCF_HELDOUT_EVAL",
        "depends_on": [calibration_id, dependency],
        "resource": "gpu",
        "priority": 31,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_mut_gcf_legacy_evaluation.sh",
        ],
        "input_manifest": "{dep_" + calibration_id + "_output}/audit.json",
        "config_files": [
            str(source_root / "frozen_candidate_manifest.json"),
            str(source_root / "matched_thresholds.json"),
        ],
        "expected_output": heldout_output,
        "required_output_files": [
            "matrix/audit.json",
            "matrix/run_manifest.json",
            "matrix/_RUN_COMPLETE.json",
            "final/final_artifact_audit.json",
            "final/run_manifest.json",
            "final/_RUN_COMPLETE.json",
        ],
        "required_log_marker": "[MUT_GCF_LEGACY_HELDOUT_PASS]",
        "environment": {
            **common,
            "ACTION": "heldout",
            "HELDOUT_CSV": str(test_csv),
            "CALIBRATION_RUN_DIR": "{dep_" + calibration_id + "_output}",
            "OURS_SCHEMA_ROOT": str(source_root / "schema_reference"),
            "OUTPUT_ROOT": "{task_output}",
            "WNODE_CACHE_DB": "{task_output}/_cache/wnode.sqlite3",
            "NODE_EMB_CACHE_DIR": "{task_output}/_cache/node_embeddings",
        },
    }
    standardized = {
        "id": "mut_gcf_legacy_standardized",
        "dataset": "mutagenicity",
        "stage": "AM_MUT_GCF_STANDARDIZED",
        "runner_dataset": "paper-cell-mutagenicity-gcfexplainer-repair",
        "runner_stage": "AM_MUT_GCF_STANDARDIZED",
        "depends_on": [heldout_id, dependency],
        "resource": "cpu",
        "priority": 32,
        "data_splits": [],
        "manifest_only": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/standardize_mut_gcf_legacy_cell.py",
            "--config",
            "configs/hpc.yaml",
            "--heldout-root",
            "{dep_" + heldout_id + "_output}",
            "--frozen-root",
            str(source_root),
            "--output-dir",
            "{task_output}",
            "--proc-root",
            "/proc",
        ],
        "input_manifest": "{dep_" + heldout_id + "_output}/final/run_manifest.json",
        "config_files": [str(source_root / "frozen_candidate_manifest.json")],
        "expected_output": standardized_output,
        "required_output_files": [
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            "table2_gcfexplainer_k10.csv",
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "summary.json",
            "run_manifest.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
            "artifact_manifest.json",
            "final_artifact_audit.json",
            "freeze_manifest.json",
            "_FINALIZED.json",
            "PASS",
        ],
        "required_log_marker": "[MUT_GCF_LEGACY_STANDARDIZATION_PASS]",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "RUN_TASTEMOLNET": "0",
        },
    }
    return [calibration, heldout, standardized]


def _threshold_task(
    *, dataset: str, source: Path, fresh_root: Path, priority: int
) -> dict[str, Any]:
    task_id = f"{dataset}_comrecgc_threshold_freeze"
    return {
        "id": task_id,
        "dataset": dataset,
        "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "runner_dataset": f"paper-threshold-{dataset}-repair",
        "runner_stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "depends_on": [],
        "resource": "cpu",
        "priority": priority,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/verify_frozen_threshold_contract.py",
            "--config",
            "configs/hpc.yaml",
            "--dataset",
            dataset,
            "--source",
            str(source),
            "--output",
            "{task_output}",
        ],
        "input_manifest": str(source),
        "expected_output": str(
            fresh_root / f"cells/{dataset}/comrecgc/threshold-freeze/attempt-{{attempt}}"
        ),
        "required_output_files": [
            "frozen_threshold_contract.json",
            "threshold_adoption_audit.json",
            "PASS",
        ],
        "required_log_marker": f"[FROZEN_THRESHOLD_CONTRACT_PASS] dataset={dataset}",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
    }


def _am_comrecgc_tasks(
    *, spec: Mapping[str, Any], sources: Mapping[str, SourceEvidence], fresh_root: Path
) -> list[dict[str, Any]]:
    shared = _mapping(_nested(spec, "am_comrecgc", "shared"), label="am_comrecgc.shared")
    upstream = _absolute_path(shared.get("upstream_root"), label="am_comrecgc.shared.upstream_root", kind="dir")
    molclr_root = _absolute_path(shared.get("molclr_root"), label="am_comrecgc.shared.molclr_root", kind="dir")
    molclr_checkpoint = _absolute_path(
        shared.get("molclr_checkpoint"),
        label="am_comrecgc.shared.molclr_checkpoint",
        kind="file",
    )
    tasks: list[dict[str, Any]] = []
    for offset, dataset in enumerate(("mutagenicity", "aids")):
        key = "mutagenicity" if dataset == "mutagenicity" else "aids"
        source_key = f"{'mut' if dataset == 'mutagenicity' else 'aids'}_comrec_generation"
        values = _mapping(_nested(spec, "am_comrecgc", key), label=f"am_comrecgc.{key}")
        dataset_dir = _absolute_path(values.get("dataset_dir"), label=f"am_comrecgc.{key}.dataset_dir", kind="dir")
        dataset_csv = _absolute_path(values.get("dataset_csv"), label=f"am_comrecgc.{key}.dataset_csv", kind="file")
        teacher = _absolute_path(values.get("teacher_path"), label=f"am_comrecgc.{key}.teacher_path", kind="file")
        distance = _absolute_path(values.get("distance_checkpoint"), label=f"am_comrecgc.{key}.distance_checkpoint", kind="file")
        threshold_source = _absolute_path(values.get("thresholds_source"), label=f"am_comrecgc.{key}.thresholds_source", kind="file")
        source_csv: Path | None = None
        if dataset == "aids":
            source_csv = _absolute_path(values.get("source_csv"), label="am_comrecgc.aids.source_csv", kind="file")
        threshold = _threshold_task(
            dataset=dataset,
            source=threshold_source,
            fresh_root=fresh_root,
            priority=40 + offset,
        )
        tasks.append(threshold)
        threshold_id = threshold["id"]
        source_dependency = SOURCE_TASK_IDS[source_key]
        environment = {
            "AUTODL_PYTHON": "{python}",
            "DATASET": dataset,
            "SOURCE_GENERATION_ROOT": str(sources[source_key].output_root),
            "COMRECGC_UPSTREAM_ROOT": str(upstream),
            "DATASET_DIR": str(dataset_dir),
            "DATASET_CSV": str(dataset_csv),
            "TEACHER_PATH": str(teacher),
            "DISTANCE_CHECKPOINT": str(distance),
            "MOLCLR_ROOT": str(molclr_root),
            "MOLCLR_CHECKPOINT": str(molclr_checkpoint),
            "THRESHOLDS_PATH": (
                "{dep_" + threshold_id + "_output}/frozen_threshold_contract.json"
            ),
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cuda:0",
            "RUN_TASTEMOLNET": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if source_csv is not None:
            environment["SOURCE_CSV"] = str(source_csv)
        tasks.append(
            {
                "id": f"{dataset}_comrecgc_standardized",
                "dataset": dataset,
                "stage": "AM_COMRECGC_HELDOUT_EVAL",
                "runner_dataset": f"paper-cell-{dataset}-comrecgc-repair",
                "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
                "depends_on": [source_dependency, threshold_id],
                "resource": "gpu",
                "priority": 50 + offset,
                "data_splits": ["test"],
                "selector_parameters_frozen": True,
                "read_only_test": True,
                "command": [
                    "bash",
                    "{project_root}/scripts/autodl/run_comrecgc_standardized_continuation.sh",
                ],
                "input_manifest": (
                    "{dep_" + source_dependency + "_output}/source_adoption.json"
                ),
                "config_files": [
                    "{dep_" + threshold_id + "_output}/frozen_threshold_contract.json"
                ],
                "expected_output": str(
                    fresh_root
                    / f"cells/{dataset}/comrecgc/standardized/attempt-{{attempt}}"
                ),
                "required_output_files": [
                    "adoption_manifest.json",
                    "standardized/_FINALIZED.json",
                    "standardized/run_manifest.json",
                    "run_manifest.json",
                    "final_gate.json",
                    "_RUN_COMPLETE.json",
                    "PASS",
                ],
                "required_log_marker": (
                    f"[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset={dataset}"
                ),
                "environment": environment,
                "semantic_failure_markers": [
                    "source_closure_changed",
                    "live_writer_detected",
                    "graph_hash_collision_or_corruption",
                    "selected comrecgc transition is not one unique",
                    "test leakage",
                ],
            }
        )
    return tasks


def _load_and_validate_spec(spec_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(spec_path).expanduser().resolve(strict=True)
    spec = _read_object(path)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RepairManifestError(f"Unsupported repair spec schema: {spec.get('schema_version')!r}")
    if spec.get("controller_id") != MANIFEST_CONTROLLER_ID:
        raise RepairManifestError(
            f"Repair controller_id must be {MANIFEST_CONTROLLER_ID!r}"
        )
    if spec.get("paper_frozen") is not True or spec.get("run_tastemolnet") != 0:
        raise RepairManifestError("Repair spec requires paper_frozen=true and run_tastemolnet=0")
    return path, spec


def build_repair_payload(
    *, spec_path: str | Path, proc_root_override: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_file, spec = _load_and_validate_spec(spec_path)
    runtime_root = _absolute_path(spec.get("runtime_root"), label="runtime_root", kind="dir")
    project_root = _absolute_path(spec.get("project_root"), label="project_root", kind="dir")
    python = _absolute_path(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError(f"Configured Python is not executable: {python}")
    fresh_root = _absolute_path(spec.get("fresh_output_root"), label="fresh_output_root", kind="fresh")
    allowed_root = (runtime_root / "outputs/autodl").resolve(strict=False)
    try:
        fresh_root.relative_to(allowed_root)
    except ValueError as exc:
        raise RepairManifestError(
            f"fresh_output_root must stay below {allowed_root}: {fresh_root}"
        ) from exc
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError(f"Repair fresh_output_root already exists: {fresh_root}")
    if "paper/" in str(fresh_root).replace("\\", "/").lower():
        raise RepairManifestError("Repair outputs may not target the paper tree")
    proc_root = (
        Path(proc_root_override).expanduser().resolve(strict=True)
        if proc_root_override is not None
        else _absolute_path(spec.get("proc_root", "/proc"), label="proc_root", kind="dir")
    )
    head = _git_head(project_root)
    required_commits = _assert_required_commits(
        project_root, spec.get("required_execution_commits", [])
    )

    source_controller = _mapping(spec.get("source_controller"), label="source_controller")
    source_manifest = _absolute_path(
        source_controller.get("manifest"), label="source_controller.manifest", kind="file"
    )
    source_controller_root = _absolute_path(
        source_controller.get("root"), label="source_controller.root", kind="dir"
    )
    sources_spec = _mapping(spec.get("sources"), label="sources")
    if set(sources_spec) != set(SOURCE_CONTROLLER_TERMINALS) | set(ARTIFACT_TERMINALS):
        raise RepairManifestError(
            "sources must contain exactly bace_b14, mut_gcf_freeze, "
            "mut_comrec_generation, and aids_comrec_generation"
        )
    sources: dict[str, SourceEvidence] = {}
    for name, expected_task in SOURCE_CONTROLLER_TERMINALS.items():
        values = _mapping(sources_spec[name], label=f"sources.{name}")
        if values.get("task_id") != expected_task:
            raise RepairManifestError(
                f"sources.{name}.task_id must be {expected_task!r}"
            )
        output_root = _absolute_path(
            values.get("output_root"), label=f"sources.{name}.output_root", kind="dir"
        )
        required = (
            ("FINAL_PASS.json", "PASS")
            if name == "bace_b14"
            else (
                "frozen_candidate_manifest.json",
                "matched_thresholds.json",
                "export/selected_top20.csv",
                "schema_reference/table2_ours_k10.csv",
                "PASS",
            )
        )
        audit = verify_controller_terminal(
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            task_id=expected_task,
            expected_output_root=output_root,
            required_files=required,
            proc_root=proc_root,
        )
        sources[name] = SourceEvidence(name, "controller_terminal", output_root, audit)
    for name, dataset in ARTIFACT_TERMINALS.items():
        values = _mapping(sources_spec[name], label=f"sources.{name}")
        output_root = _absolute_path(
            values.get("output_root"), label=f"sources.{name}.output_root", kind="dir"
        )
        audit = verify_comrecgc_generation_terminal(
            dataset=dataset,
            expected_output_root=output_root,
            proc_root=proc_root,
        )
        sources[name] = SourceEvidence(name, "artifact_terminal", output_root, audit)

    tasks: list[dict[str, Any]] = []
    for name in SOURCE_TASK_IDS:
        tasks.append(
            _source_task(
                source=sources[name],
                project_root=project_root,
                fresh_root=fresh_root,
                source_manifest=(
                    source_manifest if name in SOURCE_CONTROLLER_TERMINALS else None
                ),
                source_controller_root=(
                    source_controller_root
                    if name in SOURCE_CONTROLLER_TERMINALS
                    else None
                ),
                task_id=SOURCE_CONTROLLER_TERMINALS.get(name),
            )
        )
    checkpoint = _absolute_path(
        _nested(spec, "bace", "gnn_checkpoint"),
        label="bace.gnn_checkpoint",
        kind="dir",
    )
    tasks.append(
        _bace_ours_standardizer(
            spec=spec,
            b14_root=sources["bace_b14"].output_root,
            fresh_root=fresh_root,
            checkpoint=checkpoint,
        )
    )
    bace_comrec_tasks = _bace_comrecgc_tasks(
        spec=spec, fresh_root=fresh_root, project_root=project_root, python=python
    )
    tasks.extend(bace_comrec_tasks)
    tasks.extend(
        _mut_gcf_tasks(
            spec=spec,
            source_root=sources["mut_gcf_freeze"].output_root,
            fresh_root=fresh_root,
        )
    )
    tasks.extend(_am_comrecgc_tasks(spec=spec, sources=sources, fresh_root=fresh_root))
    task_ids = [str(task.get("id") or "") for task in tasks]
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise RepairManifestError("Repair task IDs are empty or duplicated")
    forbidden = {
        "four_by_four_main_results_export",
        "four_by_four_final_matrix_audit",
        "tastemolnet_ours",
        "tastemolnet_gcfexplainer",
        "tastemolnet_globalgce",
        "tastemolnet_comrecgc",
    }
    if forbidden.intersection(task_ids):
        raise RepairManifestError("Repair manifest contains a forbidden final/Taste task")

    payload = {
        "schema_version": 1,
        "controller_id": MANIFEST_CONTROLLER_ID,
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "auto",
            # The main v1 controller remains live.  Bound this second control
            # plane so the two schedulers cannot together flood host CPUs.
            "max_cpu_tasks": 2,
            "launch_grace_seconds": 180,
            "max_transient_retries": 1,
            "keep_alive_when_blocked": True,
        },
        "resource_gates": {
            "min_available_ram_gb": 32,
            "min_free_disk_gb": 20,
            "max_cpu_load_fraction": 0.9,
        },
        "repair_contract": {
            "schema_version": SPEC_SCHEMA,
            "spec_path": str(spec_file),
            "spec_sha256": sha256_file(spec_file),
            "execution_project_root": str(project_root),
            "execution_commit": head,
            "required_execution_commits": required_commits,
            "fresh_output_root": str(fresh_root),
            "source_controller_manifest": str(source_manifest),
            "source_controller_manifest_sha256": sha256_file(source_manifest),
            "source_controller_root": str(source_controller_root),
            "source_evidence": {
                name: dict(source.audit) for name, source in sources.items()
            },
            "shared_gpu_uuid_lock_root": str(runtime_root / "locks"),
            "old_v2_continuation_lock_inherited": False,
            "b0_b14_scientific_rerun": False,
            "taste_tasks_present": False,
            "final_audit_present": False,
        },
        "tasks": tasks,
    }
    # No continuation object is copied from the old campaign.  Both live
    # controllers coordinate only through the project-wide UUID locks.
    if "continuation" in payload:
        raise AssertionError("Repair manifest must not inherit the v2 continuation lock")
    summary = {
        "status": "PASS",
        "controller_id": MANIFEST_CONTROLLER_ID,
        "task_count": len(tasks),
        "task_ids": task_ids,
        "source_names": list(sources),
        "source_controller_id": load_controller_manifest(source_manifest).controller_id,
        "fresh_output_root": str(fresh_root),
        "execution_commit": head,
        "max_cpu_tasks": 2,
        "continuation_lock_inherited": False,
    }
    return payload, summary


def validate_repair_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="four-by-four-repair-validate-") as directory:
        path = Path(directory) / "manifest.json"
        _atomic_json(path, payload)
        manifest = load_controller_manifest(path)
    task_ids = [task.task_id for task in manifest.tasks]
    return {
        "status": "PASS",
        "controller_id": manifest.controller_id,
        "task_count": len(manifest.tasks),
        "task_ids": task_ids,
        "manifest_sha256": manifest.sha256,
        "max_cpu_tasks": int(manifest.runtime.get("max_cpu_tasks", 0)),
        "test_boundary_validated": True,
        "dependency_graph_validated": True,
    }


def build_repair_manifest(
    *, spec_path: str | Path, output_path: str | Path, proc_root_override: str | Path | None = None
) -> dict[str, Any]:
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        raise RepairManifestError("Repair manifest output must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Repair manifest output must be fresh: {destination}")
    payload, build_summary = build_repair_payload(
        spec_path=spec_path, proc_root_override=proc_root_override
    )
    validation = validate_repair_payload(payload)
    _atomic_json(destination, payload)
    try:
        frozen = load_controller_manifest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("Published manifest differs from validated bytes")
    return {
        **build_summary,
        "manifest": str(destination),
        "manifest_sha256": frozen.sha256,
        "test_boundary_validated": True,
        "dependency_graph_validated": True,
    }


__all__ = [
    "ARTIFACT_TERMINALS",
    "MANIFEST_CONTROLLER_ID",
    "RepairManifestError",
    "SOURCE_CONTROLLER_TERMINALS",
    "SOURCE_TASK_IDS",
    "SPEC_SCHEMA",
    "build_repair_manifest",
    "build_repair_payload",
    "publish_source_adoption",
    "sha256_file",
    "validate_repair_payload",
    "verify_comrecgc_generation_terminal",
    "verify_controller_terminal",
]
