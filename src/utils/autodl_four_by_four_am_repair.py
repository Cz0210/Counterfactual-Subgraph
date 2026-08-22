"""Build the AIDS/Mutagenicity-only AutoDL ComRecGC repair controller.

This continuation is deliberately narrower than the original four-by-four
repair.  It adopts exactly four PASS terminals from
``four_methods_four_datasets_repair_v1`` (the two recovered-generation
adoptions and the two frozen-threshold terminals) and schedules only two fresh
held-out ComRecGC standardization jobs.  Scientific paths are copied from the
immutable repair-v1 controller manifest and cross-checked against the adopted
PASS evidence; they are not re-entered in a second hand-maintained spec.

The builder and every runtime source-gate task require the execution checkout
to descend from the exact Git-safety fix used by
``verify_comrecgc_checkout``.  No BACE, GCFExplainer, TasteMolNet, final-export,
or predecessor-continuation task can enter the emitted graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerManifest,
    TaskSpec,
    load_controller_manifest,
)
from src.utils.autodl_four_by_four_repair import (
    RepairManifestError,
    sha256_file,
    verify_controller_terminal,
)


SPEC_SCHEMA = "four_by_four_am_repair_spec_v2"
MANIFEST_CONTROLLER_ID = "four_methods_four_datasets_am_repair_v2"
SOURCE_CONTROLLER_ID = "four_methods_four_datasets_repair_v1"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"
VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT = (
    "d8b113281d24e9340bfe2379e7451ffa8adff70a"
)
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceDefinition:
    key: str
    dataset: str
    kind: str
    task_id: str
    source_name: str | None
    gate_task_id: str
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class AdoptedSource:
    definition: SourceDefinition
    output_root: Path
    audit: Mapping[str, Any]
    semantic: Mapping[str, Any]


SOURCE_DEFINITIONS: dict[str, SourceDefinition] = {
    "mut_generation": SourceDefinition(
        key="mut_generation",
        dataset="mutagenicity",
        kind="generation_adoption",
        task_id="repair_source_mut_comrec_generation",
        source_name="mut_comrec_generation",
        gate_task_id="am_v2_source_mut_comrec_generation",
        required_files=("source_adoption.json", "PASS"),
    ),
    "mut_threshold": SourceDefinition(
        key="mut_threshold",
        dataset="mutagenicity",
        kind="threshold",
        task_id="mutagenicity_comrecgc_threshold_freeze",
        source_name=None,
        gate_task_id="am_v2_source_mut_comrec_threshold",
        required_files=(
            "frozen_threshold_contract.json",
            "threshold_adoption_audit.json",
            "PASS",
        ),
    ),
    "aids_generation": SourceDefinition(
        key="aids_generation",
        dataset="aids",
        kind="generation_adoption",
        task_id="repair_source_aids_comrec_generation",
        source_name="aids_comrec_generation",
        gate_task_id="am_v2_source_aids_comrec_generation",
        required_files=("source_adoption.json", "PASS"),
    ),
    "aids_threshold": SourceDefinition(
        key="aids_threshold",
        dataset="aids",
        kind="threshold",
        task_id="aids_comrecgc_threshold_freeze",
        source_name=None,
        gate_task_id="am_v2_source_aids_comrec_threshold",
        required_files=(
            "frozen_threshold_contract.json",
            "threshold_adoption_audit.json",
            "PASS",
        ),
    ),
}

STANDARDIZATION_TASK_IDS = {
    "mutagenicity": "mutagenicity_comrecgc_standardized",
    "aids": "aids_comrecgc_standardized",
}

STANDARDIZED_REQUIRED_FILES = (
    "adoption_manifest.json",
    "standardized/_FINALIZED.json",
    "standardized/run_manifest.json",
    "run_manifest.json",
    "final_gate.json",
    "_RUN_COMPLETE.json",
    "PASS",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: str | Path) -> dict[str, Any]:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise RepairManifestError(f"JSON source must be absolute and physical: {logical}")
    source = logical.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"JSON source must be a nonempty file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"Invalid JSON object: {source}") from exc
    if not isinstance(value, dict):
        raise RepairManifestError(f"Expected one JSON object: {source}")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairManifestError(f"{label} must be one object")
    return value


def _absolute_path(value: Any, *, label: str, kind: str) -> Path:
    logical = Path(str(value or "")).expanduser()
    if not logical.is_absolute():
        raise RepairManifestError(f"{label} must be absolute: {logical}")
    if logical.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink: {logical}")
    if kind == "fresh":
        return logical.resolve(strict=False)
    resolved = logical.resolve(strict=True)
    if kind == "dir" and not resolved.is_dir():
        raise RepairManifestError(f"{label} must be a directory: {resolved}")
    if kind == "file" and (
        not resolved.is_file() or resolved.stat().st_size <= 0
    ):
        raise RepairManifestError(f"{label} must be a nonempty file: {resolved}")
    return resolved


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


def _git_head(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError(
            f"Cannot resolve execution worktree HEAD: {project_root}"
        ) from exc
    if HEX40.fullmatch(value) is None:
        raise RepairManifestError(f"Execution HEAD is not a full commit: {value!r}")
    return value


def verify_fix_ancestry(
    *, project_root: str | Path, required_fix_commit: str
) -> dict[str, str]:
    """Require the one reviewed safe-Git fix, not an arbitrary commit token."""

    root = _absolute_path(project_root, label="project_root", kind="dir")
    if required_fix_commit != VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT:
        raise RepairManifestError(
            "verify_comrecgc_checkout_safe_git_fix_commit must equal the reviewed "
            f"fix {VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT}"
        )
    head = _git_head(root)
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                required_fix_commit,
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError("Safe-Git fix ancestry check could not run") from exc
    if completed.returncode != 0:
        raise RepairManifestError(
            "Execution worktree does not contain the reviewed "
            f"verify_comrecgc_checkout safe-Git fix: {required_fix_commit}"
        )
    return {
        "required_fix_commit": required_fix_commit,
        "execution_head": head,
        "is_ancestor": "true",
    }


def _expected_dataset_name(dataset: str) -> str:
    return "Mutagenicity" if dataset == "mutagenicity" else "AIDS"


def _validate_generation_adoption(
    *, definition: SourceDefinition, output_root: Path
) -> dict[str, Any]:
    payload = _read_object(output_root / "source_adoption.json")
    failures: list[str] = []
    if payload.get("schema_version") != "four_by_four_repair_source_adoption_v1":
        failures.append("schema_version")
    if payload.get("status") != "PASS":
        failures.append("status")
    if payload.get("source_name") != definition.source_name:
        failures.append("source_name")
    evidence = _mapping(
        payload.get("source_evidence"), label="source_adoption.source_evidence"
    )
    if evidence.get("status") != "PASS":
        failures.append("source_evidence.status")
    if evidence.get("kind") != "artifact_terminal":
        failures.append("source_evidence.kind")
    if evidence.get("dataset") != definition.dataset:
        failures.append("source_evidence.dataset")
    if evidence.get("large_payload_sha256_computed") is not False:
        failures.append("source_evidence.large_payload_sha256_computed")
    if evidence.get("payload_claimed_sha256_cross_manifest_agreement") is not True:
        failures.append("source_evidence.payload_claimed_sha256_cross_manifest_agreement")
    if evidence.get("closure_member_count") != 6:
        failures.append("source_evidence.closure_member_count")
    closure_members = evidence.get("closure_members")
    expected_members = {
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "freeze_only_recovery.json",
        "frozen_payload_closure_audit.json",
        "adoption_manifest.json",
        "counterfactuals.pt",
    }
    if not isinstance(closure_members, list) or set(closure_members) != expected_members:
        failures.append("source_evidence.closure_members")
    writer = evidence.get("live_writer_audit")
    if not isinstance(writer, Mapping) or writer.get("writable_fd_count") != 0:
        failures.append("source_evidence.live_writer_audit")
    claimed_sha = str(evidence.get("payload_claimed_sha256") or "")
    if HEX64.fullmatch(claimed_sha) is None:
        failures.append("source_evidence.payload_claimed_sha256")
    generation_root_value = evidence.get("source_output_root")
    try:
        generation_root = _absolute_path(
            generation_root_value,
            label=f"sources.{definition.key}.generation_root",
            kind="dir",
        )
    except (FileNotFoundError, RepairManifestError):
        generation_root = Path(str(generation_root_value or ""))
        failures.append("source_evidence.source_output_root")
    if failures:
        raise RepairManifestError(
            f"Repair-v1 generation adoption {definition.key} is invalid: {failures}"
        )
    return {
        "kind": definition.kind,
        "dataset": definition.dataset,
        "generation_root": str(generation_root),
        "generation_payload_claimed_sha256": claimed_sha,
        "source_adoption_sha256": sha256_file(output_root / "source_adoption.json"),
        "large_payload_sha256_computed": False,
    }


def _float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RepairManifestError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise RepairManifestError(f"{label} is not finite")
    return result


def _validate_threshold(
    *, definition: SourceDefinition, output_root: Path
) -> dict[str, Any]:
    contract_path = output_root / "frozen_threshold_contract.json"
    audit_path = output_root / "threshold_adoption_audit.json"
    contract = _read_object(contract_path)
    audit = _read_object(audit_path)
    expected_dataset = _expected_dataset_name(definition.dataset)
    failures: list[str] = []
    for field, expected in (
        ("status", "PASS"),
        ("dataset", expected_dataset),
        ("cf_mode", "strict_flip"),
        ("distance_line", DISTANCE_LINE),
        ("test_used_for_selection", False),
        ("threshold_fitted_on_test", False),
        ("selection_used_test", False),
        ("shared_across_methods", True),
    ):
        if contract.get(field) != expected:
            failures.append(f"contract.{field}")
    if contract.get("threshold_source_split") not in {
        "existing_frozen_protocol",
        "frozen_protocol",
        "legacy_frozen_protocol",
    }:
        failures.append("contract.threshold_source_split")
    raw_thresholds = contract.get("thresholds")
    try:
        thresholds = [float(value) for value in raw_thresholds]
    except (TypeError, ValueError):
        thresholds = []
    if (
        len(thresholds) != 601
        or thresholds != sorted(set(thresholds))
        or any(not math.isfinite(value) or value < 0.0 for value in thresholds)
        or not math.isclose(thresholds[0], 0.0, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(thresholds[-1], 0.0535, rel_tol=0.0, abs_tol=1e-15)
    ):
        failures.append("contract.thresholds")
    try:
        theta_star = _float(contract.get("theta_star"), label="theta_star")
        cost_cap = _float(contract.get("cost_cap"), label="cost_cap")
    except RepairManifestError:
        theta_star = cost_cap = math.nan
        failures.append("contract.theta_star_or_cost_cap")
    if not math.isclose(theta_star, 0.05, rel_tol=0.0, abs_tol=1e-15):
        failures.append("contract.theta_star")
    if not math.isclose(cost_cap, 0.0535, rel_tol=0.0, abs_tol=1e-15):
        failures.append("contract.cost_cap")
    threshold_hash = str(contract.get("threshold_config_hash") or "")
    if HEX64.fullmatch(threshold_hash) is None:
        failures.append("contract.threshold_config_hash")
    if audit.get("schema_version") != "frozen_threshold_adoption_audit_v1":
        failures.append("audit.schema_version")
    for field, expected in (
        ("status", "PASS"),
        ("dataset", expected_dataset),
        ("threshold_count", 601),
        ("test_used_for_selection", False),
        ("shared_across_methods", True),
        ("failures", []),
    ):
        if audit.get(field) != expected:
            failures.append(f"audit.{field}")
    try:
        audit_theta = _float(audit.get("theta_star"), label="audit.theta_star")
        audit_cap = _float(audit.get("cost_cap"), label="audit.cost_cap")
    except RepairManifestError:
        audit_theta = audit_cap = math.nan
        failures.append("audit.theta_star_or_cost_cap")
    if not math.isclose(audit_theta, theta_star, rel_tol=0.0, abs_tol=1e-15):
        failures.append("audit.theta_star")
    if not math.isclose(audit_cap, cost_cap, rel_tol=0.0, abs_tol=1e-15):
        failures.append("audit.cost_cap")
    source_hash = str(contract.get("source_contract_sha256") or "")
    if HEX64.fullmatch(source_hash) is None or audit.get(
        "source_contract_sha256"
    ) != source_hash:
        failures.append("source_contract_sha256")
    if contract.get("source_contract") != audit.get("source_contract"):
        failures.append("source_contract")
    if (output_root / "PASS").read_text(encoding="utf-8") != "PASS\n":
        failures.append("PASS")
    if failures:
        raise RepairManifestError(
            f"Repair-v1 threshold terminal {definition.key} is invalid: {failures}"
        )
    return {
        "kind": definition.kind,
        "dataset": definition.dataset,
        "threshold_contract": str(contract_path.resolve(strict=True)),
        "threshold_contract_sha256": sha256_file(contract_path),
        "threshold_config_hash": threshold_hash,
        "threshold_count": len(thresholds),
        "theta_star": theta_star,
        "cost_cap": cost_cap,
        "test_used_for_selection": False,
    }


def _source_controller(
    *, source_manifest: Path, source_controller_root: Path, control_root: Path
) -> ControllerManifest:
    expected_namespace = control_root / SOURCE_NAMESPACE
    expected_manifest_parent = expected_namespace / "manifests"
    if source_manifest.parent != expected_manifest_parent.resolve(strict=True):
        raise RepairManifestError(
            "repair-v1 source manifest must use the configured continuation namespace"
        )
    expected_root = expected_namespace / SOURCE_CONTROLLER_ID
    if source_controller_root != expected_root.resolve(strict=True):
        raise RepairManifestError("repair-v1 source controller root is not exact")
    manifest = load_controller_manifest(source_manifest)
    if manifest.controller_id != SOURCE_CONTROLLER_ID:
        raise RepairManifestError(
            f"Source controller must be {SOURCE_CONTROLLER_ID}, got {manifest.controller_id}"
        )
    return manifest


def verify_repair_v1_source(
    *,
    source_key: str,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    control_root: str | Path,
    expected_output_root: str | Path,
    project_root: str | Path,
    required_fix_commit: str,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Verify one exact repair-v1 PASS terminal and its AM semantics."""

    if source_key not in SOURCE_DEFINITIONS:
        raise RepairManifestError(f"Unknown AM repair source: {source_key}")
    definition = SOURCE_DEFINITIONS[source_key]
    manifest_path = _absolute_path(
        source_manifest, label="source_controller.manifest", kind="file"
    )
    controller_root = _absolute_path(
        source_controller_root, label="source_controller.root", kind="dir"
    )
    configured_control = _absolute_path(control_root, label="control_root", kind="dir")
    source = _source_controller(
        source_manifest=manifest_path,
        source_controller_root=controller_root,
        control_root=configured_control,
    )
    fix = verify_fix_ancestry(
        project_root=project_root, required_fix_commit=required_fix_commit
    )
    output_root = _absolute_path(
        expected_output_root,
        label=f"sources.{source_key}.output_root",
        kind="dir",
    )
    controller_audit = verify_controller_terminal(
        source_manifest=manifest_path,
        source_controller_root=controller_root,
        task_id=definition.task_id,
        expected_output_root=output_root,
        required_files=definition.required_files,
        proc_root=proc_root,
    )
    if definition.kind == "generation_adoption":
        semantic = _validate_generation_adoption(
            definition=definition, output_root=output_root
        )
    else:
        semantic = _validate_threshold(definition=definition, output_root=output_root)
    return {
        "schema_version": "four_by_four_am_repair_source_terminal_v2",
        "status": "PASS",
        "source_key": source_key,
        "dataset": definition.dataset,
        "kind": definition.kind,
        "source_controller_id": source.controller_id,
        "controller_terminal": controller_audit,
        "semantic": semantic,
        "execution_fix_gate": fix,
        "verified_at": _utc_now(),
    }


def publish_source_gate(
    *, source_key: str, evidence: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    if source_key not in SOURCE_DEFINITIONS:
        raise RepairManifestError(f"Unknown AM repair source: {source_key}")
    if evidence.get("status") != "PASS" or evidence.get("source_key") != source_key:
        raise RepairManifestError("Cannot publish a mismatched/non-PASS source gate")
    destination = _absolute_path(output_dir, label="source gate output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Source gate output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o755)
    payload = {
        "schema_version": "four_by_four_am_repair_source_gate_v2",
        "status": "PASS",
        "source_key": source_key,
        "evidence": dict(evidence),
        "published_at": _utc_now(),
    }
    _atomic_json(destination / "source_gate.json", payload)
    descriptor = os.open(
        destination / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _source_gate_task(
    *,
    source: AdoptedSource,
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    fresh_root: Path,
    required_fix_commit: str,
    priority: int,
) -> dict[str, Any]:
    definition = source.definition
    command = [
        "{python}",
        "{project_root}/scripts/autodl/build_four_by_four_am_repair_manifest.py",
        "--config",
        "configs/hpc.yaml",
        "verify-source",
        "--source-key",
        definition.key,
        "--source-manifest",
        str(source_manifest),
        "--source-controller-root",
        str(source_controller_root),
        "--control-root",
        str(control_root),
        "--expected-output-root",
        str(source.output_root),
        "--project-root",
        "{project_root}",
        "--required-fix-commit",
        required_fix_commit,
        "--output-dir",
        "{task_output}",
    ]
    is_threshold = definition.kind == "threshold"
    return {
        "id": definition.gate_task_id,
        "dataset": definition.dataset if is_threshold else "repair-source-audit",
        "stage": (
            "AM_COMRECGC_THRESHOLD_FREEZE"
            if is_threshold
            else "FOUR_BY_FOUR_AM_REPAIR_SOURCE_ADOPTION"
        ),
        "runner_dataset": f"am-v2-source-{definition.key}",
        "runner_stage": "FOUR_BY_FOUR_AM_REPAIR_SOURCE_GATE",
        "depends_on": [],
        "resource": "cpu",
        "priority": priority,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": is_threshold,
        "command": command,
        "input_manifest": str(
            source.output_root
            / (
                "frozen_threshold_contract.json"
                if is_threshold
                else "source_adoption.json"
            )
        ),
        "config_files": [
            str(source.output_root / name)
            for name in definition.required_files
            if name != "PASS"
        ],
        "expected_output": str(
            fresh_root
            / f"source-adoptions/{definition.key}/attempt-{{attempt}}"
        ),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": (
            "[FOUR_BY_FOUR_AM_REPAIR_SOURCE_GATE_PASS] "
            f"source={definition.key}"
        ),
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RUN_TASTEMOLNET": "0",
        },
    }


def _require_source_standardization_task(
    *,
    manifest: ControllerManifest,
    dataset: str,
    generation: AdoptedSource,
    threshold: AdoptedSource,
) -> TaskSpec:
    task_id = STANDARDIZATION_TASK_IDS[dataset]
    if task_id not in manifest.by_id:
        raise RepairManifestError(f"repair-v1 manifest has no task {task_id}")
    task = manifest.by_id[task_id]
    expected_dependencies = {
        generation.definition.task_id,
        threshold.definition.task_id,
    }
    failures: list[str] = []
    if task.dataset != dataset:
        failures.append("dataset")
    if task.stage != "AM_COMRECGC_HELDOUT_EVAL":
        failures.append("stage")
    if task.resource != "gpu":
        failures.append("resource")
    if set(task.depends_on) != expected_dependencies:
        failures.append("depends_on")
    if task.data_splits != ("test",):
        failures.append("data_splits")
    if not task.selector_parameters_frozen:
        failures.append("selector_parameters_frozen")
    if not task.read_only_test:
        failures.append("read_only_test")
    if task.command is None or not any(
        value.endswith("/scripts/autodl/run_comrecgc_standardized_continuation.sh")
        for value in task.command
    ):
        failures.append("command")
    if not set(STANDARDIZED_REQUIRED_FILES).issubset(task.required_output_files):
        failures.append("required_output_files")
    environment = task.environment
    if str(environment.get("DATASET")) != dataset:
        failures.append("environment.DATASET")
    generation_root = Path(str(generation.semantic["generation_root"])).resolve(
        strict=True
    )
    try:
        source_generation = Path(
            str(environment.get("SOURCE_GENERATION_ROOT") or "")
        ).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError):
        source_generation = Path("/")
    if source_generation != generation_root:
        failures.append("environment.SOURCE_GENERATION_ROOT")
    if str(environment.get("RUN_TASTEMOLNET")) != "0":
        failures.append("environment.RUN_TASTEMOLNET")
    if dataset == "aids" and not environment.get("SOURCE_CSV"):
        failures.append("environment.SOURCE_CSV")
    if dataset == "mutagenicity" and environment.get("SOURCE_CSV"):
        failures.append("environment.SOURCE_CSV")
    if failures:
        raise RepairManifestError(
            f"repair-v1 {task_id} scientific contract is invalid: {failures}"
        )
    return task


def _scientific_environment(
    *, task: TaskSpec, dataset: str, threshold_contract: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    directory_fields = (
        "SOURCE_GENERATION_ROOT",
        "COMRECGC_UPSTREAM_ROOT",
        "DATASET_DIR",
        "MOLCLR_ROOT",
    )
    file_fields = (
        "DATASET_CSV",
        "TEACHER_PATH",
        "DISTANCE_CHECKPOINT",
        "MOLCLR_CHECKPOINT",
    )
    if dataset == "aids":
        file_fields = (*file_fields, "SOURCE_CSV")
    copied: dict[str, str] = {"DATASET": dataset}
    for field in directory_fields:
        copied[field] = str(
            _absolute_path(
                task.environment.get(field),
                label=f"{task.task_id}.environment.{field}",
                kind="dir",
            )
        )
    for field in file_fields:
        copied[field] = str(
            _absolute_path(
                task.environment.get(field),
                label=f"{task.task_id}.environment.{field}",
                kind="file",
            )
        )
    copied.update(
        {
            "AUTODL_PYTHON": "{python}",
            "THRESHOLDS_PATH": str(threshold_contract),
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cuda:0",
            "RUN_TASTEMOLNET": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if dataset == "aids":
        copied["THETA_STAR"] = str(task.environment.get("THETA_STAR", "0.05"))
        copied["COST_CAP"] = str(task.environment.get("COST_CAP", "0.0535"))
        if not math.isclose(float(copied["THETA_STAR"]), 0.05, abs_tol=1e-15):
            raise RepairManifestError("repair-v1 AIDS THETA_STAR is not 0.05")
        if not math.isclose(float(copied["COST_CAP"]), 0.0535, abs_tol=1e-15):
            raise RepairManifestError("repair-v1 AIDS COST_CAP is not 0.0535")
    scientific = {
        key: value
        for key, value in copied.items()
        if key
        not in {
            "AUTODL_PYTHON",
            "OUTPUT_ROOT",
            "DEVICE",
            "PYTHONDONTWRITEBYTECODE",
        }
    }
    digest = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return copied, {
        "source_task_id": task.task_id,
        "scientific_environment": scientific,
        "scientific_environment_sha256": digest,
    }


def _standardized_task(
    *,
    dataset: str,
    source_task: TaskSpec,
    generation: AdoptedSource,
    threshold: AdoptedSource,
    fresh_root: Path,
    priority: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold_contract = Path(str(threshold.semantic["threshold_contract"]))
    environment, provenance = _scientific_environment(
        task=source_task,
        dataset=dataset,
        threshold_contract=threshold_contract,
    )
    generation_gate = generation.definition.gate_task_id
    threshold_gate = threshold.definition.gate_task_id
    task = {
        "id": STANDARDIZATION_TASK_IDS[dataset],
        "dataset": dataset,
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": f"paper-cell-{dataset}-comrecgc-am-repair-v2",
        "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
        "depends_on": [generation_gate, threshold_gate],
        "resource": "gpu",
        "priority": priority,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_comrecgc_standardized_continuation.sh",
        ],
        "input_manifest": "{dep_" + generation_gate + "_output}/source_gate.json",
        "config_files": [
            "{dep_" + threshold_gate + "_output}/source_gate.json",
            str(threshold_contract),
        ],
        "expected_output": str(
            fresh_root
            / f"cells/{dataset}/comrecgc/standardized/attempt-{{attempt}}"
        ),
        "required_output_files": list(STANDARDIZED_REQUIRED_FILES),
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
            "dubious ownership",
        ],
    }
    return task, provenance


def build_am_repair_fragment(
    *,
    sources: Mapping[str, AdoptedSource],
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    fresh_root: Path,
    required_fix_commit: str,
    manifest: ControllerManifest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact four source gates and two fresh scientific tasks."""

    if set(sources) != set(SOURCE_DEFINITIONS):
        raise RepairManifestError("AM repair fragment requires exactly four sources")
    tasks = [
        _source_gate_task(
            source=sources[key],
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            fresh_root=fresh_root,
            required_fix_commit=required_fix_commit,
            priority=index + 1,
        )
        for index, key in enumerate(SOURCE_DEFINITIONS)
    ]
    scientific_provenance: dict[str, Any] = {}
    for offset, (dataset, generation_key, threshold_key) in enumerate(
        (
            ("mutagenicity", "mut_generation", "mut_threshold"),
            ("aids", "aids_generation", "aids_threshold"),
        )
    ):
        generation = sources[generation_key]
        threshold = sources[threshold_key]
        source_task = _require_source_standardization_task(
            manifest=manifest,
            dataset=dataset,
            generation=generation,
            threshold=threshold,
        )
        task, provenance = _standardized_task(
            dataset=dataset,
            source_task=source_task,
            generation=generation,
            threshold=threshold,
            fresh_root=fresh_root,
            priority=20 + offset,
        )
        tasks.append(task)
        scientific_provenance[dataset] = provenance
    expected_ids = {
        *(definition.gate_task_id for definition in SOURCE_DEFINITIONS.values()),
        *STANDARDIZATION_TASK_IDS.values(),
    }
    task_ids = {str(task.get("id") or "") for task in tasks}
    if len(tasks) != 6 or task_ids != expected_ids:
        raise RepairManifestError("AM repair fragment escaped its six-task boundary")
    return tasks, scientific_provenance


def _load_spec(spec_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _absolute_path(spec_path, label="spec", kind="file")
    spec = _read_object(path)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RepairManifestError(
            f"Unsupported AM repair spec schema: {spec.get('schema_version')!r}"
        )
    if spec.get("controller_id") != MANIFEST_CONTROLLER_ID:
        raise RepairManifestError(
            f"AM repair controller_id must be {MANIFEST_CONTROLLER_ID!r}"
        )
    if spec.get("paper_frozen") is not True or spec.get("run_tastemolnet") != 0:
        raise RepairManifestError(
            "AM repair requires paper_frozen=true and run_tastemolnet=0"
        )
    forbidden_top_level = {"continuation", "bace", "gcf", "taste", "final"}
    if forbidden_top_level.intersection(spec):
        raise RepairManifestError("AM repair spec contains a forbidden route/guard")
    return path, spec


def build_am_repair_payload(
    *, spec_path: str | Path, proc_root_override: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_file, spec = _load_spec(spec_path)
    runtime_root = _absolute_path(spec.get("runtime_root"), label="runtime_root", kind="dir")
    control_root = _absolute_path(spec.get("control_root"), label="control_root", kind="dir")
    project_root = _absolute_path(spec.get("project_root"), label="project_root", kind="dir")
    python = _absolute_path(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError(f"Configured Python is not executable: {python}")
    fresh_root = _absolute_path(
        spec.get("fresh_output_root"), label="fresh_output_root", kind="fresh"
    )
    allowed_output_root = (runtime_root / "outputs/autodl").resolve(strict=False)
    try:
        fresh_root.relative_to(allowed_output_root)
    except ValueError as exc:
        raise RepairManifestError(
            f"fresh_output_root must stay below {allowed_output_root}: {fresh_root}"
        ) from exc
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError(f"AM repair fresh_output_root already exists: {fresh_root}")
    expected_controller_root = (
        control_root / SOURCE_NAMESPACE / MANIFEST_CONTROLLER_ID
    ).resolve(strict=False)
    if expected_controller_root.exists() or expected_controller_root.is_symlink():
        raise RepairManifestError(
            f"AM repair controller root already exists: {expected_controller_root}"
        )
    if "paper/" in str(fresh_root).replace("\\", "/").lower():
        raise RepairManifestError("AM repair outputs may not target the paper tree")
    proc_root = (
        _absolute_path(proc_root_override, label="proc_root", kind="dir")
        if proc_root_override is not None
        else _absolute_path(spec.get("proc_root", "/proc"), label="proc_root", kind="dir")
    )
    required_fix_commit = str(
        spec.get("verify_comrecgc_checkout_safe_git_fix_commit") or ""
    )
    fix_gate = verify_fix_ancestry(
        project_root=project_root, required_fix_commit=required_fix_commit
    )
    source_controller = _mapping(spec.get("source_controller"), label="source_controller")
    source_manifest = _absolute_path(
        source_controller.get("manifest"),
        label="source_controller.manifest",
        kind="file",
    )
    source_controller_root = _absolute_path(
        source_controller.get("root"), label="source_controller.root", kind="dir"
    )
    manifest = _source_controller(
        source_manifest=source_manifest,
        source_controller_root=source_controller_root,
        control_root=control_root,
    )
    sources_spec = _mapping(spec.get("sources"), label="sources")
    if set(sources_spec) != set(SOURCE_DEFINITIONS):
        raise RepairManifestError(
            "sources must contain exactly mut_generation, mut_threshold, "
            "aids_generation, and aids_threshold"
        )
    sources: dict[str, AdoptedSource] = {}
    for key, definition in SOURCE_DEFINITIONS.items():
        values = _mapping(sources_spec[key], label=f"sources.{key}")
        if set(values) != {"task_id", "output_root"}:
            raise RepairManifestError(
                f"sources.{key} must contain only task_id and output_root"
            )
        if values.get("task_id") != definition.task_id:
            raise RepairManifestError(
                f"sources.{key}.task_id must be {definition.task_id!r}"
            )
        output_root = _absolute_path(
            values.get("output_root"),
            label=f"sources.{key}.output_root",
            kind="dir",
        )
        audit = verify_repair_v1_source(
            source_key=key,
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            expected_output_root=output_root,
            project_root=project_root,
            required_fix_commit=required_fix_commit,
            proc_root=proc_root,
        )
        sources[key] = AdoptedSource(
            definition=definition,
            output_root=output_root,
            audit=audit,
            semantic=_mapping(audit["semantic"], label=f"sources.{key}.semantic"),
        )
    tasks, scientific_provenance = build_am_repair_fragment(
        sources=sources,
        source_manifest=source_manifest,
        source_controller_root=source_controller_root,
        control_root=control_root,
        fresh_root=fresh_root,
        required_fix_commit=required_fix_commit,
        manifest=manifest,
    )
    payload: dict[str, Any] = {
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
        "am_repair_contract": {
            "schema_version": SPEC_SCHEMA,
            "spec_path": str(spec_file),
            "spec_sha256": sha256_file(spec_file),
            "execution_project_root": str(project_root),
            "execution_commit": fix_gate["execution_head"],
            "verify_comrecgc_checkout_safe_git_fix_commit": required_fix_commit,
            "verify_fix_is_ancestor": True,
            "fresh_output_root": str(fresh_root),
            "expected_controller_root": str(expected_controller_root),
            "expected_manifest_path": str(
                control_root
                / SOURCE_NAMESPACE
                / "manifests"
                / f"{MANIFEST_CONTROLLER_ID}.json"
            ),
            "source_controller_id": manifest.controller_id,
            "source_controller_manifest": str(source_manifest),
            "source_controller_manifest_sha256": manifest.sha256,
            "source_controller_root": str(source_controller_root),
            "source_evidence": {key: dict(source.audit) for key, source in sources.items()},
            "scientific_environment_adopted_from_repair_v1": scientific_provenance,
            "shared_gpu_uuid_lock_root": str(runtime_root / "locks"),
            "old_continuation_guard_inherited": False,
            "bace_tasks_present": False,
            "gcf_tasks_present": False,
            "taste_tasks_present": False,
            "final_export_tasks_present": False,
        },
        "tasks": tasks,
    }
    if "continuation" in payload:
        raise AssertionError("AM repair may not inherit an old continuation guard")
    validation = validate_am_repair_payload(payload)
    summary = {
        "status": "PASS",
        "controller_id": MANIFEST_CONTROLLER_ID,
        "task_count": validation["task_count"],
        "task_ids": validation["task_ids"],
        "source_controller_id": manifest.controller_id,
        "fresh_output_root": str(fresh_root),
        "execution_commit": fix_gate["execution_head"],
        "required_fix_commit": required_fix_commit,
        "max_cpu_tasks": 2,
        "continuation_guard_inherited": False,
    }
    return payload, summary


def validate_am_repair_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="four-by-four-am-repair-validate-") as directory:
        manifest_path = Path(directory) / "manifest.json"
        _atomic_json(manifest_path, payload)
        manifest = load_controller_manifest(manifest_path)
    expected_ids = {
        *(definition.gate_task_id for definition in SOURCE_DEFINITIONS.values()),
        *STANDARDIZATION_TASK_IDS.values(),
    }
    task_ids = [task.task_id for task in manifest.tasks]
    if len(task_ids) != 6 or set(task_ids) != expected_ids:
        raise RepairManifestError("AM repair manifest is not the exact six-task graph")
    if "continuation" in payload:
        raise RepairManifestError("AM repair manifest inherited an old guard")
    forbidden = re.compile(r"(?:^|[_-])(bace|gcf(?:explainer)?|taste(?:molnet)?)(?:$|[_-])")
    for task in manifest.tasks:
        if forbidden.search(task.task_id.lower()) or forbidden.search(task.stage.lower()):
            raise RepairManifestError(f"Forbidden AM repair task: {task.task_id}")
        if task.dataset not in {"aids", "mutagenicity", "repair-source-audit"}:
            raise RepairManifestError(f"Forbidden AM repair dataset: {task.dataset}")
    if int(manifest.runtime.get("max_cpu_tasks", 0)) != 2:
        raise RepairManifestError("AM repair max_cpu_tasks must equal two")
    if int(manifest.runtime.get("max_gpus", 0)) != 4:
        raise RepairManifestError("AM repair must share the four-GPU scheduler ceiling")
    return {
        "status": "PASS",
        "controller_id": manifest.controller_id,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "manifest_sha256": manifest.sha256,
        "max_cpu_tasks": 2,
        "test_boundary_validated": True,
        "dependency_graph_validated": True,
    }


def build_am_repair_manifest(
    *,
    spec_path: str | Path,
    output_path: str | Path,
    proc_root_override: str | Path | None = None,
) -> dict[str, Any]:
    destination = _absolute_path(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"AM repair manifest output must be fresh: {destination}")
    payload, summary = build_am_repair_payload(
        spec_path=spec_path, proc_root_override=proc_root_override
    )
    expected = Path(
        str(payload["am_repair_contract"]["expected_manifest_path"])
    ).resolve(strict=False)
    if destination != expected:
        raise RepairManifestError(
            f"AM repair manifest path must be exact: expected={expected}, actual={destination}"
        )
    validation = validate_am_repair_payload(payload)
    _atomic_json(destination, payload)
    try:
        frozen = load_controller_manifest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("Published AM repair manifest differs from validated bytes")
    return {
        **summary,
        "manifest": str(destination),
        "manifest_sha256": frozen.sha256,
        "test_boundary_validated": True,
        "dependency_graph_validated": True,
    }


__all__ = [
    "MANIFEST_CONTROLLER_ID",
    "SOURCE_CONTROLLER_ID",
    "SOURCE_DEFINITIONS",
    "SPEC_SCHEMA",
    "VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT",
    "build_am_repair_fragment",
    "build_am_repair_manifest",
    "build_am_repair_payload",
    "publish_source_gate",
    "validate_am_repair_payload",
    "verify_fix_ancestry",
    "verify_repair_v1_source",
]
