"""Fail-closed Mutagenicity trace-off parity continuation for AutoDL.

The historical 50k COMRECGC generation was produced with the action trace
enabled.  Mutagenicity's chemistry contract requires a *distinct* trace-off
reference before that payload may enter the standardized paper cell.  This
module builds a persistent controller which:

1. revalidates the exact immutable trace-on source;
2. waits for an exact, SHA-pinned, memory-bounded AIDS repair terminal to pass;
3. runs one fresh, checkpointed, trace-disabled 50k replay on an exclusive
   GPU with the same data order, models, seed, and scientific parameters;
4. compares the fresh reference to the immutable trace-on payload with
   :func:`assert_trace_parity`;
5. adopts the already-completed repair-v2 common-recourse stage read-only;
6. resumes at chemistry in a fresh CPU-only standardization root.

No helper in this module writes below any historical source root.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerManifest,
    load_controller_manifest,
)
from scripts.verify_comrecgc_checkout import verify_checkout
from src.baselines.comrecgc.contracts import (
    GenerationParameters,
    UPSTREAM_COMMIT,
    ordered_ids_sha256,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.baselines.comrecgc.generation_checkpoint import (
    COMPLETE_FILENAME,
    LATEST_FILENAME,
    LATEST_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MIRRORED_FILENAME,
    SQLITE_FILENAME,
    STATE_FILENAME,
    list_generation_checkpoints,
    scientific_command_sha256,
    validate_generation_checkpoint,
)
from src.baselines.comrecgc.graph_trace import assert_trace_parity
from src.eval.am_legacy_standardization import scan_live_writers
from src.utils.autodl_four_by_four_am_repair import (
    SOURCE_DEFINITIONS,
    VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
    verify_fix_ancestry,
    verify_repair_v1_source,
)
from src.utils.autodl_four_by_four_repair import (
    RepairManifestError,
    verify_controller_terminal,
)


SPEC_SCHEMA = "mut_comrecgc_traceoff_parity_spec_v1"
CONTROLLER_ID = "four_methods_four_datasets_mut_traceoff_parity_v1"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"

TRACE_SOURCE_TASK_ID = "mut_trace_on_source_gate"
THRESHOLD_TASK_ID = "mut_threshold_source_gate"
AIDS_WAIT_TASK_ID = "mut_wait_aids_comrecgc_pass"
INSTRUMENTATION_EQUIVALENCE_TASK_ID = (
    "mut_checkpoint_instrumentation_equivalence"
)
TRACEOFF_TASK_ID = "mut_traceoff_reference_50k"
PARITY_TASK_ID = "mut_assert_trace_on_off_parity"
COMMON_TASK_ID = "mut_adopt_repair_v2_common_recourse"
STANDARDIZE_TASK_ID = "mut_standardize_from_parity_common"

SOURCE_PROJECT_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
# This historical child of SOURCE_PROJECT_COMMIT is the reviewed release that
# introduced completed-step checkpoint/resume instrumentation.  The controller
# itself runs from a newer immutable worktree, but the scientific 50k replay is
# deliberately executed from this separate, SHA-pinned worktree.
INSTRUMENTATION_PROJECT_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
LEGACY_SOURCE_INVENTORY_SHA256 = (
    "240db0f3bfe6c02ef7e60798d7e6ae40c9494d2aae8befe5f687bdda4324c390"
)
INSTRUMENTATION_SOURCE_INVENTORY_SHA256 = (
    "6b3f509ff01059e54006053981c1f8914eacba2bbfd42c3787f9566c626ff1c6"
)
SOURCE_UPSTREAM_COMMIT = UPSTREAM_COMMIT
SOURCE_CONFIG_SHA256 = "5a6088a56741627cc7353d5450999d90aad18076304060c7621ea6c0cad11f34"
SOURCE_DATASET_SHA256 = "6fd22a03193e772a36b608ce05e858dc76cf125f0a25c2779728cb44ccf445dd"
SOURCE_PARENT_ORDER_SHA256 = "2c5d6842cbb2f74d72b5d4ec281a59a1f327c2b4959059fa20fb78bf5974e573"
SOURCE_GNN_SHA256 = "22045e5a6a833d6ed980cef9834859859136a1e2f644d19d78bd63345585f239"
SOURCE_DISTANCE_SHA256 = "bc64c16340c9170388ff1b3951d2ee4cb9a372456b09691ecd6bb2a881f17648"
SOURCE_PAYLOAD_SHA256 = "fc790056e3c3267153ac3e2d707717ccec88a89e4d0ad3b677af82d5a90cd3d3"
SOURCE_CANDIDATE_COUNT = 100_235
SOURCE_PARENT_COUNT = 1_448
SOURCE_STEPS = 50_000
SOURCE_SEED = 0
SOURCE_BATCH_SIZE = 128
INSTRUMENTATION_EQUIVALENCE_STEPS = 500
AIDS_MINIMUM_HEADROOM_BYTES = 400 * 1024**3
AIDS_V4_MINIMUM_HEADROOM_BYTES = 128 * 1024**3

INSTRUMENTATION_SOURCE_FILES = (
    "scripts/baselines/comrecgc/run_generation.py",
    "src/baselines/comrecgc/runtime.py",
    "src/baselines/comrecgc/graph_trace.py",
    "src/baselines/comrecgc/generation_checkpoint.py",
    "src/baselines/comrecgc/generation_loop.py",
    "src/baselines/comrecgc/live_graph_state.py",
    "src/baselines/comrecgc/transition_cache.py",
    "src/baselines/comrecgc/storage_guard.py",
    "src/baselines/comrecgc/contracts.py",
    "src/baselines/comrecgc/project_dataset.py",
    "src/baselines/comrecgc/model_adapter.py",
    "src/baselines/comrecgc/upstream.py",
)

SOURCE_PARAMETERS = {
    "candidate_capacity": 100_000,
    "heads": 5,
    "sample_size": 10_000,
    "seed": SOURCE_SEED,
    "steps": SOURCE_STEPS,
    "teleport": 0.1,
    "theta": 0.1,
}

AIDS_CONTROLLER_ID = "four_methods_four_datasets_aids_comrecgc_repair_v3"
AIDS_TASK_ID = "aids_comrecgc_standardized_cpu_highmem"
REPAIR_V2_CONTROLLER_ID = "four_methods_four_datasets_am_repair_v2"
REPAIR_V2_MUT_TASK_ID = "mutagenicity_comrecgc_standardized"
REPAIR_V2_COMMON_REQUIRED = (
    "_RUN_COMPLETE.json",
    "run_manifest.json",
    "selected_common_recourses.csv",
    "selected_common_recourses.json",
    "representative_counterfactuals.pt",
)

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairManifestError(f"{label} must be one object")
    return value


def _read_object(path: str | Path) -> dict[str, Any]:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise RepairManifestError(f"JSON path must be absolute and physical: {logical}")
    source = logical.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"JSON path must be a nonempty file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"Invalid JSON object: {source}") from exc
    if not isinstance(value, dict):
        raise RepairManifestError(f"Expected one JSON object: {source}")
    return value


def _absolute(value: Any, *, label: str, kind: str) -> Path:
    logical = Path(str(value or "")).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise RepairManifestError(f"{label} must be absolute and physical: {logical}")
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


def _publish_fresh_gate(
    *, output_dir: str | Path, filename: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    output = _absolute(output_dir, label="gate output", kind="fresh")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"gate output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755)
    value = dict(payload)
    _atomic_json(output / filename, value)
    descriptor = os.open(output / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


def _git_head(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError(f"Cannot resolve Git HEAD: {project_root}") from exc
    if HEX40.fullmatch(value) is None:
        raise RepairManifestError(f"Execution Git HEAD is malformed: {value!r}")
    return value


def _git_is_ancestor(*, ancestor: str, descendant: str, project_root: Path) -> bool:
    if HEX40.fullmatch(ancestor) is None or HEX40.fullmatch(descendant) is None:
        raise RepairManifestError("Git ancestry requires two full commit IDs")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError("Cannot verify instrumentation ancestry") from exc
    if result.returncode not in {0, 1}:
        raise RepairManifestError(
            "Cannot verify instrumentation ancestry: " + result.stderr.strip()
        )
    return result.returncode == 0


def _require_clean_tracked_worktree(project_root: Path, *, label: str) -> None:
    try:
        value = subprocess.check_output(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError(f"Cannot audit {label} worktree") from exc
    if value.strip():
        raise RepairManifestError(f"{label} worktree has modified tracked files")


def instrumentation_source_inventory(project_root: str | Path) -> dict[str, Any]:
    """Freeze the exact source/AST identity used by one equivalence arm."""

    root = _absolute(project_root, label="instrumentation source root", kind="dir")
    files: dict[str, Any] = {}
    for relative in INSTRUMENTATION_SOURCE_FILES:
        logical_path = root / relative
        if not logical_path.exists():
            files[relative] = {"present": False}
            continue
        source_path = _absolute(
            logical_path, label=f"instrumentation source {relative}", kind="file"
        )
        source = source_path.read_bytes()
        try:
            tree = ast.parse(source.decode("utf-8"), filename=str(source_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise RepairManifestError(
                f"Cannot parse instrumentation source: {source_path}"
            ) from exc
        definitions: dict[str, str] = {}
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                definitions[node.name] = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode("utf-8")
                ).hexdigest()
        files[relative] = {
            "present": True,
            "sha256": hashlib.sha256(source).hexdigest(),
            "top_level_definition_ast_sha256": definitions,
        }
    payload = {
        "schema_version": "mut_checkpoint_instrumentation_source_inventory_v1",
        "project_root": str(root),
        "project_commit": _git_head(root),
        "files": files,
    }
    payload["inventory_sha256"] = stable_json_sha256(
        {key: value for key, value in payload.items() if key != "project_root"}
    )
    return payload


def expected_traceoff_scientific_argv(
    *,
    instrumentation_project_root: str | Path,
    upstream_root: str | Path,
    dataset_dir: str | Path,
    gnn_checkpoint: str | Path,
    distance_checkpoint: str | Path,
    generation_root: str | Path,
    checkpoint_root: str | Path,
    mirror_root: str | Path,
) -> tuple[str, ...]:
    """Reproduce the reviewed 66487c0 CLI's canonical, resume-stable argv."""

    values: dict[str, Any] = {
        "batch_size": SOURCE_BATCH_SIZE,
        "checkpoint_interval_steps": 500,
        "checkpoint_keep_last": 2,
        "checkpoint_mirror_root": str(Path(mirror_root).resolve(strict=False)),
        "checkpoint_root": str(Path(checkpoint_root).resolve(strict=False)),
        "config": "configs/hpc.yaml",
        "dataset": "mutagenicity",
        "dataset_dir": str(Path(dataset_dir).resolve(strict=True)),
        "device": "cuda:0",
        "distance_checkpoint": str(Path(distance_checkpoint).resolve(strict=True)),
        "expected_cache_inventory_sha256": None,
        "gnn_checkpoint": str(Path(gnn_checkpoint).resolve(strict=True)),
        "graph_state_dir": str(
            (Path(generation_root).resolve(strict=False) / "graph_state")
        ),
        "mode": "full",
        "output_dir": str(Path(generation_root).resolve(strict=False)),
        "parent_limit": SOURCE_PARENT_COUNT,
        "parity_reference": None,
        "progress_interval_steps": 25,
        "project_root": str(Path(instrumentation_project_root).resolve(strict=True)),
        "route": "project",
        "set": ["inference.fallback_to_heuristic=false"],
        "source_csv": None,
        "storage_check_every_steps": 500,
        "storage_guard_root": str(Path(generation_root).resolve(strict=False)),
        "storage_min_free_gib": 50.0,
        "storage_min_free_inodes": 100_000,
        "storage_min_free_ratio": 0.02,
        "trace_output_dir": None,
        "trusted_dataset_payload": None,
        "upstream_root": str(Path(upstream_root).resolve(strict=True)),
    }
    return (
        "scripts/baselines/comrecgc/run_generation.py",
        *(
            f"--{key.replace('_', '-')}="
            + json.dumps(value, sort_keys=True, separators=(",", ":"))
            for key, value in sorted(values.items())
        ),
    )


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Mut trace parity requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _source_files(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "run_manifest.json",
        "resolved_config.json",
        "_RUN_COMPLETE.json",
        "freeze_only_recovery.json",
        "frozen_payload_closure_audit.json",
        "adoption_manifest.json",
        "trace/trace_summary.json",
        "trace/_TRACE_COMPLETE.json",
        "trace/candidate_action_lineage.json",
        "trace/candidate_action_lineage_index.jsonl",
        "trace/selected_action_trace_manifest.json",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        source = _absolute(root / name, label=f"trace source {name}", kind="file")
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RepairManifestError(f"trace source file escapes root: {source}") from exc
        result[name] = {
            "path": str(source),
            "size": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    return result


def verify_traced_source(
    *, source_root: str | Path, proc_root: str | Path = "/proc", hash_payload: bool
) -> dict[str, Any]:
    """Verify the exact 50k trace-on source without modifying it."""

    root = _absolute(source_root, label="trace-on source root", kind="dir")
    files = _source_files(root)
    resolved = _read_object(root / "resolved_config.json")
    manifest = _read_object(root / "run_manifest.json")
    complete = _read_object(root / "_RUN_COMPLETE.json")
    recovery = _read_object(root / "freeze_only_recovery.json")
    closure = _read_object(root / "frozen_payload_closure_audit.json")
    trace = _read_object(root / "trace/trace_summary.json")
    trace_complete = _read_object(root / "trace/_TRACE_COMPLETE.json")
    failures: list[str] = []
    expected_scalar = {
        "dataset": "mutagenicity",
        "route": "project_adapted",
        "mode": "full",
        "project_commit": SOURCE_PROJECT_COMMIT,
        "upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "parent_limit": SOURCE_PARENT_COUNT,
        "config_sha256": SOURCE_CONFIG_SHA256,
        "generation_parent_ids_sha256": SOURCE_PARENT_ORDER_SHA256,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for label, value in (("resolved", resolved), ("manifest", manifest)):
        for key, expected in expected_scalar.items():
            if value.get(key) != expected:
                failures.append(f"{label}.{key}")
        if value.get("parameters") != SOURCE_PARAMETERS:
            failures.append(f"{label}.parameters")
        if len(value.get("generation_parent_ids") or ()) != SOURCE_PARENT_COUNT:
            failures.append(f"{label}.generation_parent_ids")
        elif ordered_ids_sha256(value["generation_parent_ids"]) != SOURCE_PARENT_ORDER_SHA256:
            failures.append(f"{label}.generation_parent_ids_content")
        dataset_audit = _mapping(
            value.get("dataset_audit"), label=f"{label}.dataset_audit"
        )
        if dataset_audit.get("dataset_fingerprint") != SOURCE_DATASET_SHA256:
            failures.append(f"{label}.dataset_audit.dataset_fingerprint")
        if (
            dataset_audit.get("generation_parent_ids_sha256")
            != SOURCE_PARENT_ORDER_SHA256
        ):
            failures.append(
                f"{label}.dataset_audit.generation_parent_ids_sha256"
            )
        gnn = _mapping(value.get("gnn"), label=f"{label}.gnn")
        distance = _mapping(value.get("distance_model"), label=f"{label}.distance")
        if gnn.get("checkpoint_sha256") != SOURCE_GNN_SHA256:
            failures.append(f"{label}.gnn.checkpoint_sha256")
        if distance.get("checkpoint_sha256") != SOURCE_DISTANCE_SHA256:
            failures.append(f"{label}.distance.checkpoint_sha256")
    expected_config_sha = stable_json_sha256(
        {key: value for key, value in resolved.items() if key != "config_sha256"}
    )
    if resolved.get("config_sha256") != SOURCE_CONFIG_SHA256:
        failures.append("resolved.config_sha256_claim")
    if expected_config_sha != SOURCE_CONFIG_SHA256:
        failures.append("resolved.config_sha256_content")
    if manifest.get("config_sha256") != SOURCE_CONFIG_SHA256:
        failures.append("manifest.config_sha256")
    if manifest.get("run_complete") is not True or manifest.get("trace_enabled") is not True:
        failures.append("manifest.completion_or_trace")
    if int(manifest.get("counterfactual_candidate_count", -1)) != SOURCE_CANDIDATE_COUNT:
        failures.append("manifest.counterfactual_candidate_count")
    if manifest.get("counterfactuals_sha256") != SOURCE_PAYLOAD_SHA256:
        failures.append("manifest.counterfactuals_sha256")
    if complete.get("run_complete") is not True or complete.get(
        "counterfactuals_sha256"
    ) != SOURCE_PAYLOAD_SHA256:
        failures.append("completion")
    if recovery.get("recovery_completed") is not True or int(
        recovery.get("completed_steps", -1)
    ) != SOURCE_STEPS:
        failures.append("freeze_only_recovery")
    if closure.get("closure_complete") is not True or closure.get(
        "post_write_reload_verified"
    ) is not True:
        failures.append("frozen_payload_closure")
    if (
        trace.get("trace_only") is not True
        or int(trace.get("candidate_count", -1)) != SOURCE_CANDIDATE_COUNT
        or int(trace.get("candidate_lineage_resolved_count", -1))
        != SOURCE_CANDIDATE_COUNT
        or trace.get("algorithm_rerun") is not False
    ):
        failures.append("trace_summary")
    if trace_complete.get("trace_complete") is not True:
        failures.append("trace_complete")
    payload = _absolute(root / "counterfactuals.pt", label="trace-on payload", kind="file")
    if payload.parent != root:
        failures.append("payload_outside_source")
    if failures:
        raise RepairManifestError(f"Mut trace-on source identity mismatch: {failures}")
    writer_audit = scan_live_writers(root, proc_root=proc_root)
    actual_payload_sha = sha256_file(payload) if hash_payload else None
    if actual_payload_sha is not None and actual_payload_sha != SOURCE_PAYLOAD_SHA256:
        raise RepairManifestError("Mut trace-on payload SHA256 mismatch")
    return {
        "schema_version": "mut_trace_on_source_evidence_v1",
        "status": "PASS",
        "dataset": "mutagenicity",
        "source_root": str(root),
        "source_project_commit": SOURCE_PROJECT_COMMIT,
        "source_upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_dataset_sha256": SOURCE_DATASET_SHA256,
        "source_parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
        "source_parent_count": SOURCE_PARENT_COUNT,
        "source_parameters": dict(SOURCE_PARAMETERS),
        "source_candidate_count": SOURCE_CANDIDATE_COUNT,
        "source_payload": str(payload),
        "source_payload_claimed_sha256": SOURCE_PAYLOAD_SHA256,
        "source_payload_actual_sha256": actual_payload_sha,
        "source_payload_hashed": bool(hash_payload),
        "source_gnn_sha256": SOURCE_GNN_SHA256,
        "source_distance_sha256": SOURCE_DISTANCE_SHA256,
        "trace_enabled": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "source_files": files,
        "live_writer_audit": writer_audit,
        "verified_at": _utc_now(),
    }


def publish_traced_source_gate(
    *, source_root: str | Path, output_dir: str | Path, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    evidence = verify_traced_source(
        source_root=source_root, proc_root=proc_root, hash_payload=True
    )
    return _publish_fresh_gate(
        output_dir=output_dir,
        filename="source_gate.json",
        payload={
            "schema_version": "mut_trace_on_source_gate_v1",
            "status": "PASS",
            "evidence": evidence,
            "published_at": _utc_now(),
        },
    )


def publish_threshold_source_gate(
    *,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    control_root: str | Path,
    expected_output_root: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    evidence = verify_repair_v1_source(
        source_key="mut_threshold",
        source_manifest=source_manifest,
        source_controller_root=source_controller_root,
        control_root=control_root,
        expected_output_root=expected_output_root,
        project_root=project_root,
        required_fix_commit=VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
        proc_root=proc_root,
    )
    return _publish_fresh_gate(
        output_dir=output_dir,
        filename="source_gate.json",
        payload={
            "schema_version": "mut_threshold_source_gate_v1",
            "status": "PASS",
            "evidence": evidence,
            "published_at": _utc_now(),
        },
    )


def _validate_common_recourse_source(
    *, repair_v2_output: str | Path, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    failed_root = _absolute(
        repair_v2_output, label="repair-v2 Mut output", kind="dir"
    )
    failure = _read_object(failed_root / "FAILED.json")
    state = _read_object(failed_root / "stage_state.json")
    common = _absolute(
        failed_root / "common_recourse",
        label="repair-v2 common-recourse root",
        kind="dir",
    )
    manifest = _read_object(common / "run_manifest.json")
    complete = _read_object(common / "_RUN_COMPLETE.json")
    failures: list[str] = []
    if (
        failure.get("status") != "FAILED"
        or failure.get("dataset") != "mutagenicity"
        or failure.get("error_class") != "CalledProcessError"
        or "audit_mutagenicity_chemistry.py" not in str(failure.get("message") or "")
    ):
        failures.append("repair_v2_failure_identity")
    chemistry_failure = _read_object(failed_root / "chemistry/failure_summary.json")
    if (
        chemistry_failure.get("error_class") != "ValueError"
        or chemistry_failure.get("message")
        != "Chemistry repair cannot be frozen before trace parity passes."
        or chemistry_failure.get("test_loaded") is not False
    ):
        failures.append("trace_parity_blocker")
    if state.get("stage") != "chemistry":
        failures.append("repair_v2_stage")
    expected = {
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "route": "project_adapted",
        "mode": "full",
        "cf_mode": "strict_flip",
        "run_complete": True,
        "test_loaded": False,
        "calibration_loaded": False,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "execution_status": "FULL_EXECUTION_PASS",
        "counterfactuals_sha256": SOURCE_PAYLOAD_SHA256,
        "model_counterfactual_candidate_count": 50_620,
        "common_recourse_count": 100,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            failures.append(f"common_manifest.{key}")
    if complete.get("run_complete") is not True:
        failures.append("common_complete")
    if manifest.get("parameters") != {
        "cf_size": 100_000,
        "cluster_size": 3,
        "delta": 0.02,
        "recourse_size": 100,
        "seed": 0,
        "theta": 0.1,
    }:
        failures.append("common_manifest.parameters")
    if failures:
        raise RepairManifestError(
            f"repair-v2 Mut common-recourse is not adoptable: {failures}"
        )
    files: dict[str, dict[str, Any]] = {}
    for name in REPAIR_V2_COMMON_REQUIRED:
        path = _absolute(common / name, label=f"common {name}", kind="file")
        files[name] = {
            "path": str(path),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    writer_audit = scan_live_writers(failed_root, proc_root=proc_root)
    return {
        "schema_version": "mut_common_recourse_adoption_evidence_v1",
        "status": "PASS",
        "scientific_rerun": False,
        "common_recourse_adopted": True,
        "source_failed_root": str(failed_root),
        "source_common_recourse_root": str(common),
        "source_failure_sha256": sha256_file(failed_root / "FAILED.json"),
        "source_files": files,
        "source_manifest": manifest,
        "live_writer_audit": writer_audit,
        "verified_at": _utc_now(),
    }


def _validate_parity_gate(path: str | Path) -> dict[str, Any]:
    gate_path = _absolute(path, label="trace parity gate", kind="file")
    gate = _read_object(gate_path)
    failures: list[str] = []
    if gate.get("schema_version") != "mut_trace_on_off_parity_v1":
        failures.append("schema_version")
    if gate.get("status") != "PASS" or gate.get("trace_parity_passed") is not True:
        failures.append("status")
    if int(gate.get("candidate_count", -1)) != SOURCE_CANDIDATE_COUNT:
        failures.append("candidate_count")
    if gate.get("reference_trace_enabled") is not False:
        failures.append("reference_trace_enabled")
    if gate.get("traced_source_trace_enabled") is not True:
        failures.append("traced_source_trace_enabled")
    for key, expected in {
        "self_comparison": False,
        "trace_fields_stripped": False,
        "reference_generation_rerun": True,
        "source_generation_rerun": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "traced_payload_sha256": SOURCE_PAYLOAD_SHA256,
    }.items():
        if gate.get(key) != expected:
            failures.append(key)
    if gate.get("reference_payload_sha256") == gate.get("traced_payload_sha256"):
        # Different runs may coincidentally serialize to identical bytes, but
        # their physical roots and run manifests must remain distinct.  Byte
        # equality is not used as evidence of independence.
        pass
    reference_root = _absolute(
        gate.get("reference_root"), label="parity reference root", kind="dir"
    )
    traced_root = _absolute(
        gate.get("traced_source_root"), label="parity traced root", kind="dir"
    )
    if reference_root == traced_root:
        failures.append("self_comparison_root")
    reference_evidence = gate.get("reference_evidence")
    if not isinstance(reference_evidence, Mapping):
        failures.append("reference_evidence")
    else:
        checkpoint = reference_evidence.get("checkpoint_evidence")
        for key, expected in {
            "status": "PASS",
            "reference_root": str(reference_root),
            "reference_project_commit": INSTRUMENTATION_PROJECT_COMMIT,
            "source_algorithm_commit": SOURCE_PROJECT_COMMIT,
            "source_upstream_commit": SOURCE_UPSTREAM_COMMIT,
            "source_dataset_sha256": SOURCE_DATASET_SHA256,
            "source_parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
            "source_parameters": SOURCE_PARAMETERS,
            "reference_trace_enabled": False,
            "reference_generation_rerun": True,
            "calibration_loaded": False,
            "test_loaded": False,
        }.items():
            if reference_evidence.get(key) != expected:
                failures.append(f"reference_evidence.{key}")
        if (
            not isinstance(checkpoint, Mapping)
            or int(checkpoint.get("completed_step", -1)) != SOURCE_STEPS
            or not checkpoint.get("checkpoint_digest")
        ):
            failures.append("reference_evidence.checkpoint")
        reference_payload = _absolute(
            reference_evidence.get("reference_payload"),
            label="parity reference payload",
            kind="file",
        )
        if (
            reference_payload.parent != reference_root
            or sha256_file(reference_payload)
            != reference_evidence.get("reference_payload_sha256")
            or gate.get("reference_payload_sha256")
            != reference_evidence.get("reference_payload_sha256")
        ):
            failures.append("reference_evidence.payload")
    traced_payload = _absolute(
        gate.get("traced_payload"), label="parity traced payload", kind="file"
    )
    if (
        traced_payload.parent != traced_root
        or sha256_file(traced_payload) != SOURCE_PAYLOAD_SHA256
    ):
        failures.append("traced_payload")
    if failures:
        raise RepairManifestError(f"Mut trace parity gate is invalid: {failures}")
    return {**gate, "path": str(gate_path), "sha256": sha256_file(gate_path)}


def validate_instrumentation_equivalence_gate(
    *,
    gate_path: str | Path,
    expected_legacy_inventory_sha256: str,
    expected_instrumentation_inventory_sha256: str,
) -> dict[str, Any]:
    """Revalidate the independent 500-step gate before the 50k replay."""

    path = _absolute(
        gate_path, label="checkpoint instrumentation equivalence gate", kind="file"
    )
    gate = _read_object(path)
    source_audit = _mapping(gate.get("source_audit"), label="source_audit")
    payload_equivalence = _mapping(
        gate.get("payload_equivalence"), label="payload_equivalence"
    )
    candidate_parity = _mapping(
        payload_equivalence.get("candidate_parity"),
        label="payload_equivalence.candidate_parity",
    )
    legacy = _mapping(source_audit.get("legacy"), label="source_audit.legacy")
    instrumented = _mapping(
        source_audit.get("instrumented"), label="source_audit.instrumented"
    )
    delta = _mapping(source_audit.get("delta_audit"), label="source_audit.delta")
    failures: list[str] = []
    expected = {
        "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
        "status": "PASS",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": INSTRUMENTATION_EQUIVALENCE_STEPS,
        "seed": SOURCE_SEED,
        "source_algorithm_commit": SOURCE_PROJECT_COMMIT,
        "execution_instrumentation_commit": INSTRUMENTATION_PROJECT_COMMIT,
        "step_action_trace_exact": True,
        "rng_state_exact": True,
        "checkpoint_mirror_verified": True,
        "checkpoint_resume_exercised": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            failures.append(key)
    if gate.get("failures") != []:
        failures.append("failures")
    if (
        payload_equivalence.get("failures") != []
        or candidate_parity.get("trace_parity_passed") is not True
    ):
        failures.append("payload_equivalence")
    expected_summary = stable_json_sha256(
        {key: value for key, value in gate.items() if key != "summary_sha256"}
    )
    if gate.get("summary_sha256") != expected_summary:
        failures.append("summary_sha256")
    if delta.get("status") != "PASS" or delta.get("failures") != []:
        failures.append("source_delta")
    if legacy.get("project_commit") != SOURCE_PROJECT_COMMIT or legacy.get(
        "inventory_sha256"
    ) != str(expected_legacy_inventory_sha256):
        failures.append("legacy_source_inventory")
    if instrumented.get(
        "project_commit"
    ) != INSTRUMENTATION_PROJECT_COMMIT or instrumented.get(
        "inventory_sha256"
    ) != str(expected_instrumentation_inventory_sha256):
        failures.append("instrumentation_source_inventory")
    pass_path = path.parent / "PASS"
    if (
        path.name != "equivalence.json"
        or not pass_path.is_file()
        or pass_path.is_symlink()
        or pass_path.read_text(encoding="utf-8") != "PASS\n"
    ):
        failures.append("pass_marker")
    if failures:
        raise RepairManifestError(
            f"Mut checkpoint instrumentation gate is invalid: {failures}"
        )
    return {**gate, "path": str(path), "sha256": sha256_file(path)}


def publish_common_adoption_gate(
    *,
    repair_v2_output: str | Path,
    parity_gate: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    parity = _validate_parity_gate(parity_gate)
    evidence = _validate_common_recourse_source(
        repair_v2_output=repair_v2_output, proc_root=proc_root
    )
    return _publish_fresh_gate(
        output_dir=output_dir,
        filename="common_recourse_adoption.json",
        payload={
            "schema_version": "mut_common_recourse_adoption_gate_v1",
            "status": "PASS",
            "evidence": evidence,
            "trace_parity_path": parity["path"],
            "trace_parity_sha256": parity["sha256"],
            "trace_parity_passed": True,
            "published_at": _utc_now(),
        },
    )


def _verify_mirror_marker(
    *, local_checkpoint: Path, mirror_checkpoint: Path
) -> dict[str, Any]:
    local = validate_generation_checkpoint(
        local_checkpoint, expected_completed_step=SOURCE_STEPS
    )
    mirror = validate_generation_checkpoint(
        mirror_checkpoint,
        expected_provenance=local.provenance_fingerprints,
        expected_scientific_argv=local.scientific_argv,
        expected_command_sha256=local.command_sha256,
        expected_total_steps=SOURCE_STEPS,
        expected_completed_step=SOURCE_STEPS,
    )
    if mirror.checkpoint_digest != local.checkpoint_digest:
        raise RepairManifestError("Trace-off checkpoint mirror digest mismatch")
    expected_marker = {
        "schema_version": "comrecgc_generation_checkpoint_mirror_v1",
        "checkpoint_mirrored": True,
        "completed_step": SOURCE_STEPS,
        "checkpoint_digest": local.checkpoint_digest,
        "source_checkpoint": str(local.checkpoint_dir),
        "mirror_checkpoint": str(mirror.checkpoint_dir),
    }
    marker_files = (
        local.checkpoint_dir / MIRRORED_FILENAME,
        mirror.checkpoint_dir / MIRRORED_FILENAME,
    )
    markers: list[dict[str, Any]] = []
    for path in marker_files:
        marker = _read_object(path)
        for key, expected in expected_marker.items():
            if marker.get(key) != expected:
                raise RepairManifestError(
                    f"Trace-off checkpoint mirror proof mismatch: {path}:{key}"
                )
        markers.append(marker)
    if markers[0] != markers[1]:
        raise RepairManifestError("Trace-off local/mirror proof markers differ")
    return {
        "completed_step": SOURCE_STEPS,
        "checkpoint_digest": local.checkpoint_digest,
        "scientific_argv": list(local.scientific_argv),
        "command_sha256": local.command_sha256,
        "local_checkpoint": str(local.checkpoint_dir),
        "mirror_checkpoint": str(mirror.checkpoint_dir),
        "local_marker_sha256": sha256_file(marker_files[0]),
        "mirror_marker_sha256": sha256_file(marker_files[1]),
    }


def verify_traceoff_reference(
    *,
    reference_root: str | Path,
    traced_source_root: str | Path,
    expected_project_commit: str,
    expected_scientific_command_sha256: str,
    checkpoint_root: str | Path,
    mirror_root: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Verify that the reference is a real, independent trace-off 50k run."""

    reference = _absolute(reference_root, label="trace-off reference", kind="dir")
    traced = _absolute(traced_source_root, label="trace-on source", kind="dir")
    if reference == traced:
        raise RepairManifestError("Trace-off reference aliases the traced source root")
    expected_commit = str(expected_project_commit or "")
    if HEX40.fullmatch(expected_commit) is None:
        raise RepairManifestError("Trace-off execution commit is malformed")
    if expected_commit != INSTRUMENTATION_PROJECT_COMMIT:
        raise RepairManifestError(
            "Trace-off execution is not the reviewed checkpoint instrumentation commit"
        )
    expected_command_sha = str(expected_scientific_command_sha256 or "")
    if HEX64.fullmatch(expected_command_sha) is None:
        raise RepairManifestError("Trace-off scientific command SHA256 is malformed")
    resolved = _read_object(reference / "resolved_config.json")
    manifest = _read_object(reference / "run_manifest.json")
    complete = _read_object(reference / "_RUN_COMPLETE.json")
    payload = _absolute(
        reference / "counterfactuals.pt", label="trace-off payload", kind="file"
    )
    traced_payload = _absolute(
        traced / "counterfactuals.pt", label="trace-on payload", kind="file"
    )
    if payload.samefile(traced_payload):
        raise RepairManifestError("Trace-off payload aliases the traced source payload")
    failures: list[str] = []
    expected_scalars = {
        "dataset": "mutagenicity",
        "route": "project_adapted",
        "mode": "full",
        "project_commit": expected_commit,
        "upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "parent_limit": SOURCE_PARENT_COUNT,
        "generation_parent_ids_sha256": SOURCE_PARENT_ORDER_SHA256,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for label, value in (("resolved", resolved), ("manifest", manifest)):
        for key, expected in expected_scalars.items():
            if value.get(key) != expected:
                failures.append(f"{label}.{key}")
        parent_ids = value.get("generation_parent_ids")
        if not isinstance(parent_ids, list) or len(parent_ids) != SOURCE_PARENT_COUNT:
            failures.append(f"{label}.generation_parent_ids")
        elif ordered_ids_sha256(parent_ids) != SOURCE_PARENT_ORDER_SHA256:
            failures.append(f"{label}.generation_parent_ids_content")
        if value.get("parameters") != SOURCE_PARAMETERS:
            failures.append(f"{label}.parameters")
        dataset_audit = _mapping(
            value.get("dataset_audit"), label=f"{label}.dataset_audit"
        )
        if dataset_audit.get("dataset_fingerprint") != SOURCE_DATASET_SHA256:
            failures.append(f"{label}.dataset_fingerprint")
        if (
            dataset_audit.get("generation_parent_ids_sha256")
            != SOURCE_PARENT_ORDER_SHA256
        ):
            failures.append(f"{label}.dataset_parent_order")
        gnn = _mapping(value.get("gnn"), label=f"{label}.gnn")
        distance = _mapping(value.get("distance_model"), label=f"{label}.distance")
        if gnn.get("checkpoint_sha256") != SOURCE_GNN_SHA256:
            failures.append(f"{label}.gnn")
        if distance.get("checkpoint_sha256") != SOURCE_DISTANCE_SHA256:
            failures.append(f"{label}.distance")
    config_hash = stable_json_sha256(
        {key: value for key, value in resolved.items() if key != "config_sha256"}
    )
    if resolved.get("config_sha256") != config_hash:
        failures.append("resolved.config_sha256")
    if manifest.get("config_sha256") != config_hash:
        failures.append("manifest.config_sha256")
    resolved_argv = tuple(str(value) for value in resolved.get("scientific_argv") or ())
    if (
        not resolved_argv
        or scientific_command_sha256(resolved_argv) != expected_command_sha
        or resolved.get("command_sha256") != expected_command_sha
        or manifest.get("scientific_argv") != list(resolved_argv)
        or manifest.get("command_sha256") != expected_command_sha
    ):
        failures.append("scientific_argv")
    if manifest.get("run_complete") is not True:
        failures.append("manifest.run_complete")
    if manifest.get("algorithm_rerun") is not True:
        failures.append("manifest.algorithm_rerun")
    if manifest.get("trace_enabled") is not False:
        failures.append("manifest.trace_enabled")
    if manifest.get("trace_summary") is not None:
        failures.append("manifest.trace_summary")
    if manifest.get("trace_parity") is not None:
        failures.append("manifest.trace_parity")
    if int(manifest.get("counterfactual_candidate_count", -1)) <= 0:
        failures.append("manifest.counterfactual_candidate_count")
    if manifest.get("generation_resume_supported") is not True:
        failures.append("manifest.generation_resume_supported")
    if "project_runtime_action_trace_only_v1" in (
        manifest.get("official_compatibility_patches") or ()
    ):
        failures.append("manifest.trace_patch_present")
    actual_payload_sha = sha256_file(payload)
    if manifest.get("counterfactuals_sha256") != actual_payload_sha:
        failures.append("manifest.counterfactuals_sha256")
    if complete != {
        "run_complete": True,
        "counterfactuals_sha256": actual_payload_sha,
    }:
        failures.append("completion")
    local_root = _absolute(
        checkpoint_root, label="trace-off checkpoint root", kind="dir"
    )
    persistent_mirror = _absolute(
        mirror_root, label="trace-off checkpoint mirror", kind="dir"
    )
    if Path(str(manifest.get("generation_checkpoint_root") or "")).resolve(
        strict=True
    ) != local_root:
        failures.append("manifest.generation_checkpoint_root")
    if Path(
        str(manifest.get("generation_checkpoint_mirror_root") or "")
    ).resolve(strict=True) != persistent_mirror:
        failures.append("manifest.generation_checkpoint_mirror_root")
    if failures:
        raise RepairManifestError(
            f"Trace-off reference provenance mismatch: {failures}"
        )
    checkpoint_evidence = _verify_mirror_marker(
        local_checkpoint=local_root,
        mirror_checkpoint=persistent_mirror,
    )
    if checkpoint_evidence.get("command_sha256") != expected_command_sha:
        raise RepairManifestError("Trace-off checkpoint scientific argv changed")
    writer_audit = scan_live_writers(reference, proc_root=proc_root)
    return {
        "schema_version": "mut_traceoff_reference_evidence_v1",
        "status": "PASS",
        "reference_root": str(reference),
        "reference_project_commit": expected_commit,
        "source_algorithm_commit": SOURCE_PROJECT_COMMIT,
        "source_upstream_commit": SOURCE_UPSTREAM_COMMIT,
        "source_dataset_sha256": SOURCE_DATASET_SHA256,
        "source_parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
        "source_parameters": dict(SOURCE_PARAMETERS),
        "reference_trace_enabled": False,
        "reference_generation_rerun": True,
        "reference_payload": str(payload),
        "reference_payload_sha256": actual_payload_sha,
        "reference_candidate_count": int(
            manifest["counterfactual_candidate_count"]
        ),
        "reference_config_sha256": config_hash,
        "scientific_argv": list(resolved_argv),
        "scientific_command_sha256": expected_command_sha,
        "checkpoint_evidence": checkpoint_evidence,
        "calibration_loaded": False,
        "test_loaded": False,
        "live_writer_audit": writer_audit,
        "verified_at": _utc_now(),
    }


def publish_traceoff_reference_gate(
    *,
    reference_root: str | Path,
    traced_source_root: str | Path,
    expected_project_commit: str,
    expected_scientific_command_sha256: str,
    checkpoint_root: str | Path,
    mirror_root: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    evidence = verify_traceoff_reference(
        reference_root=reference_root,
        traced_source_root=traced_source_root,
        expected_project_commit=expected_project_commit,
        expected_scientific_command_sha256=expected_scientific_command_sha256,
        checkpoint_root=checkpoint_root,
        mirror_root=mirror_root,
        proc_root=proc_root,
    )
    return _publish_fresh_gate(
        output_dir=output_dir,
        filename="traceoff_reference.json",
        payload={
            "schema_version": "mut_traceoff_reference_gate_v1",
            "status": "PASS",
            "evidence": evidence,
            "published_at": _utc_now(),
        },
    )


def assert_mut_trace_parity(
    *,
    reference_root: str | Path,
    traced_source_root: str | Path,
    expected_project_commit: str,
    expected_scientific_command_sha256: str,
    checkpoint_root: str | Path,
    mirror_root: str | Path,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Compare one fresh trace-disabled run against the frozen trace-on source."""

    reference = _absolute(reference_root, label="trace-off reference", kind="dir")
    traced = _absolute(traced_source_root, label="trace-on source", kind="dir")
    if reference == traced:
        raise RepairManifestError("Trace parity may not compare a source to itself")
    source_evidence = verify_traced_source(
        source_root=traced, proc_root=proc_root, hash_payload=True
    )
    reference_evidence = verify_traceoff_reference(
        reference_root=reference,
        traced_source_root=traced,
        expected_project_commit=expected_project_commit,
        expected_scientific_command_sha256=expected_scientific_command_sha256,
        checkpoint_root=checkpoint_root,
        mirror_root=mirror_root,
        proc_root=proc_root,
    )
    reference_payload_path = _absolute(
        reference / "counterfactuals.pt",
        label="trace-off reference payload",
        kind="file",
    )
    traced_payload_path = _absolute(
        traced / "counterfactuals.pt", label="trace-on source payload", kind="file"
    )
    if reference_payload_path.samefile(traced_payload_path):
        raise RepairManifestError("Trace parity reference aliases the traced source payload")
    reference_payload = _torch_load(reference_payload_path)
    traced_payload = _torch_load(traced_payload_path)
    parity = assert_trace_parity(reference_payload, traced_payload)
    reference_sha = sha256_file(reference_payload_path)
    output = _absolute(output_dir, label="parity output", kind="fresh")
    payload = {
        "schema_version": "mut_trace_on_off_parity_v1",
        "status": "PASS",
        **parity,
        "reference_root": str(reference),
        "reference_run_manifest_sha256": sha256_file(reference / "run_manifest.json"),
        "reference_payload": str(reference_payload_path),
        "reference_payload_sha256": reference_sha,
        "reference_trace_enabled": False,
        "traced_source_root": str(traced),
        "traced_source_run_manifest_sha256": source_evidence["source_files"][
            "run_manifest.json"
        ]["sha256"],
        "traced_payload": str(traced_payload_path),
        "traced_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "traced_source_trace_enabled": True,
        "self_comparison": False,
        "trace_fields_stripped": False,
        "reference_generation_rerun": True,
        "source_generation_rerun": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "reference_evidence": reference_evidence,
        "compared_at": _utc_now(),
    }
    if payload.get("trace_parity_passed") is not True:
        raise RepairManifestError("assert_trace_parity did not publish PASS")
    return _publish_fresh_gate(
        output_dir=output,
        filename="trace_parity.json",
        payload=payload,
    )


def _aids_manifest(
    *,
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    expected_controller_id: str,
    expected_task_id: str,
    expected_wrapper: str,
    expected_manifest_sha256: str,
    expected_highmem_lock: Path | None = None,
    expected_flock_bin: Path | None = None,
    expected_cgroup_root: Path | None = None,
    expected_min_free_bytes: int | None = None,
    expected_proc_root: Path | None = None,
) -> ControllerManifest:
    if expected_controller_id == AIDS_CONTROLLER_ID:
        raise RepairManifestError("Known OOM-failed AIDS repair-v3 cannot release Mut")
    if (
        not expected_controller_id.startswith(
            "four_methods_four_datasets_aids_comrecgc_repair_"
        )
        or not expected_task_id.startswith("aids_comrecgc_")
        or Path(expected_wrapper).name != expected_wrapper
        or not expected_wrapper.endswith(".sh")
    ):
        raise RepairManifestError("AIDS dependency identifiers are not fail-closed")
    expected_namespace = control_root / SOURCE_NAMESPACE
    if source_manifest.parent != (expected_namespace / "manifests").resolve(strict=True):
        raise RepairManifestError("AIDS dependency manifest is outside the exact namespace")
    if source_controller_root != (expected_namespace / expected_controller_id).resolve(
        strict=True
    ):
        raise RepairManifestError("AIDS dependency controller root is not exact")
    manifest = load_controller_manifest(source_manifest)
    if manifest.sha256 != expected_manifest_sha256:
        raise RepairManifestError("AIDS dependency manifest SHA256 changed")
    if (
        manifest.controller_id != expected_controller_id
        or expected_task_id not in manifest.by_id
    ):
        raise RepairManifestError("AIDS dependency is not the exact reviewed task")
    task = manifest.by_id[expected_task_id]
    if task.resource != "cpu" or task.environment.get("GPU_REQUIRED") != "0":
        raise RepairManifestError("AIDS dependency lost its CPU-only contract")
    lock_path = (
        expected_highmem_lock
        if expected_highmem_lock is not None
        else (control_root.parent / "locks/comrecgc_common_recourse_highmem.lock")
    ).resolve(strict=False)
    if (
        tuple(task.command or ())
        != (
            "bash",
            "{project_root}/scripts/autodl/" + expected_wrapper,
        )
        or task.environment.get("COMRECGC_HIGHMEM_LOCK_PATH") != str(lock_path)
        or task.environment.get("DEVICE") != "cpu"
        or task.environment.get("CUDA_VISIBLE_DEVICES") != ""
        or not task.environment.get("COMRECGC_CGROUP_MEMORY_ROOT")
        or int(task.environment.get("COMRECGC_MIN_CGROUP_FREE_BYTES", "0")) <= 0
    ):
        raise RepairManifestError("AIDS dependency lost its shared high-memory contract")
    exact_optional = {
        "COMRECGC_FLOCK_BIN": expected_flock_bin,
        "COMRECGC_CGROUP_MEMORY_ROOT": expected_cgroup_root,
        "COMRECGC_PROC_ROOT": expected_proc_root,
    }
    for key, expected in exact_optional.items():
        if expected is not None and task.environment.get(key) != str(expected):
            raise RepairManifestError(f"AIDS/Mut shared contract differs at {key}")
    if expected_min_free_bytes is not None and int(
        task.environment.get("COMRECGC_MIN_CGROUP_FREE_BYTES", "0")
    ) != int(expected_min_free_bytes):
        raise RepairManifestError("AIDS/Mut shared cgroup headroom differs")
    return manifest


def _repair_v2_manifest(
    *,
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    expected_output_root: Path,
) -> dict[str, Any]:
    expected_namespace = control_root / SOURCE_NAMESPACE
    if source_manifest.parent != (expected_namespace / "manifests").resolve(
        strict=True
    ):
        raise RepairManifestError("repair-v2 manifest is outside the exact namespace")
    expected_controller = (expected_namespace / REPAIR_V2_CONTROLLER_ID).resolve(
        strict=True
    )
    if source_controller_root != expected_controller:
        raise RepairManifestError("repair-v2 controller root is not exact")
    manifest = load_controller_manifest(source_manifest)
    snapshot_path = _absolute(
        source_controller_root / "controller_manifest.json",
        label="repair-v2 controller snapshot",
        kind="file",
    )
    snapshot = _read_object(snapshot_path)
    if (
        manifest.controller_id != REPAIR_V2_CONTROLLER_ID
        or snapshot.get("controller_id") != manifest.controller_id
        or Path(str(snapshot.get("source_manifest") or "")).resolve(strict=True)
        != source_manifest
        or snapshot.get("source_manifest_sha256") != manifest.sha256
        or REPAIR_V2_MUT_TASK_ID not in manifest.by_id
    ):
        raise RepairManifestError("repair-v2 controller snapshot identity mismatch")
    task = manifest.by_id[REPAIR_V2_MUT_TASK_ID]
    expected_template = str(task.expected_output or "")
    expected_attempt0 = Path(expected_template.format(attempt=0)).resolve(strict=True)
    if expected_attempt0 != expected_output_root:
        raise RepairManifestError("repair-v2 Mut expected output root changed")
    task_root = source_controller_root / "tasks" / REPAIR_V2_MUT_TASK_ID
    state_path = _absolute(task_root / "state.json", label="repair-v2 state", kind="file")
    gate_path = _absolute(task_root / "gate.json", label="repair-v2 gate", kind="file")
    state = _read_object(state_path)
    gate = _read_object(gate_path)
    if state.get("state") != "FAILED" or gate.get("status") != "FAILED":
        raise RepairManifestError("repair-v2 Mut task is not the exact FAILED terminal")
    instances = _mapping(state.get("instances"), label="repair-v2 state.instances")
    matching_instances = [
        value
        for value in instances.values()
        if isinstance(value, Mapping)
        and value.get("state") == "FAILED"
        and Path(str(value.get("expected_output") or "")).resolve(strict=True)
        == expected_output_root
    ]
    runs = gate.get("runs")
    matching_runs = [
        value
        for value in (runs if isinstance(runs, list) else [])
        if isinstance(value, Mapping)
        and value.get("state") == "FAILED"
        and Path(str(value.get("expected_output") or "")).resolve(strict=True)
        == expected_output_root
    ]
    if len(matching_instances) != 1 or len(matching_runs) != 1:
        raise RepairManifestError(
            "repair-v2 Mut FAILED instance/run does not bind the exact output"
        )
    return {
        "controller_id": manifest.controller_id,
        "manifest": str(source_manifest),
        "manifest_sha256": manifest.sha256,
        "controller_root": str(source_controller_root),
        "snapshot_sha256": sha256_file(snapshot_path),
        "task_id": REPAIR_V2_MUT_TASK_ID,
        "task_state_sha256": sha256_file(state_path),
        "task_gate_sha256": sha256_file(gate_path),
        "expected_output_root": str(expected_output_root),
        "failed_instance_count": 1,
        "failed_run_count": 1,
    }


def wait_for_aids_pass(
    *,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    control_root: str | Path,
    expected_output_root: str | Path,
    expected_controller_id: str,
    expected_task_id: str,
    expected_wrapper: str,
    expected_manifest_sha256: str,
    output_dir: str | Path,
    proc_root: str | Path = "/proc",
    poll_seconds: int = 60,
) -> dict[str, Any]:
    manifest_path = _absolute(source_manifest, label="AIDS dependency manifest", kind="file")
    controller_root = _absolute(
        source_controller_root, label="AIDS dependency controller root", kind="dir"
    )
    control = _absolute(control_root, label="control root", kind="dir")
    expected = _absolute(
        expected_output_root, label="AIDS dependency expected output", kind="fresh"
    )
    manifest = _aids_manifest(
        source_manifest=manifest_path,
        source_controller_root=controller_root,
        control_root=control,
        expected_controller_id=expected_controller_id,
        expected_task_id=expected_task_id,
        expected_wrapper=expected_wrapper,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    interval = int(poll_seconds)
    if interval < 5 or interval > 300:
        raise RepairManifestError("AIDS wait poll_seconds must be in [5, 300]")
    output = _absolute(output_dir, label="AIDS wait output", kind="fresh")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"AIDS wait output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o755)
    heartbeat = output / "wait_heartbeat.json"
    while True:
        state_path = controller_root / "tasks" / expected_task_id / "state.json"
        state = _read_object(state_path)
        task_state = str(state.get("state") or "")
        _atomic_json(
            heartbeat,
            {
                "schema_version": "mut_wait_aids_comrecgc_heartbeat_v1",
                "status": "WAITING" if task_state != "PASS" else "VERIFYING",
                "source_controller_id": manifest.controller_id,
                "source_manifest_sha256": manifest.sha256,
                "source_task_id": expected_task_id,
                "observed_task_state": task_state,
                "checked_at": _utc_now(),
            },
        )
        if task_state == "PASS":
            terminal = verify_controller_terminal(
                source_manifest=manifest_path,
                source_controller_root=controller_root,
                task_id=expected_task_id,
                expected_output_root=expected,
                required_files=(
                    "standardized/_FINALIZED.json",
                    "standardized/run_manifest.json",
                    "run_manifest.json",
                    "final_gate.json",
                    "_RUN_COMPLETE.json",
                    "PASS",
                ),
                proc_root=proc_root,
            )
            payload = {
                "schema_version": "mut_wait_aids_comrecgc_pass_v1",
                "status": "PASS",
                "aids_controller_id": manifest.controller_id,
                "aids_manifest_sha256": manifest.sha256,
                "aids_task_id": expected_task_id,
                "aids_output_root": str(expected.resolve(strict=True)),
                "aids_terminal": terminal,
                "wait_completed_at": _utc_now(),
            }
            _atomic_json(output / "aids_dependency.json", payload)
            descriptor = os.open(
                output / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            try:
                os.write(descriptor, b"PASS\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return payload
        if task_state in {"FAILED", "BLOCKED", "SKIPPED"}:
            raise RepairManifestError(
                f"AIDS dependency terminal cannot release Mut replay: {task_state}"
            )
        time.sleep(interval)


def _copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    published = False
    try:
        for name in (
            STATE_FILENAME,
            SQLITE_FILENAME,
            MANIFEST_FILENAME,
            COMPLETE_FILENAME,
            MIRRORED_FILENAME,
        ):
            source_file = source / name
            if not source_file.is_file() or source_file.is_symlink():
                raise RepairManifestError(f"checkpoint mirror is incomplete: {source_file}")
            shutil.copy2(source_file, temporary / name)
        os.rename(temporary, destination)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def prepare_generation_resume(
    *, output_root: str | Path, checkpoint_root: str | Path, mirror_root: str | Path
) -> dict[str, Any]:
    """Restore a fully mirrored checkpoint only when the local authority is absent."""

    output = _absolute(output_root, label="generation output", kind="dir")
    checkpoint = _absolute(checkpoint_root, label="checkpoint root", kind="fresh")
    mirror = _absolute(mirror_root, label="checkpoint mirror", kind="dir")
    if (output / "_RUN_COMPLETE.json").is_file():
        raise RepairManifestError("Completed trace-off generation cannot be resumed")
    local_valid = list_generation_checkpoints(checkpoint) if checkpoint.is_dir() else []
    mirror_valid = list_generation_checkpoints(mirror)
    committed = []
    for source in mirror_valid:
        marker = source / MIRRORED_FILENAME
        if marker.is_file() and not marker.is_symlink():
            validation = validate_generation_checkpoint(source)
            proof = _read_object(marker)
            expected_proof = {
                "schema_version": "comrecgc_generation_checkpoint_mirror_v1",
                "checkpoint_mirrored": True,
                "completed_step": validation.completed_step,
                "checkpoint_digest": validation.checkpoint_digest,
                "mirror_checkpoint": str(validation.checkpoint_dir),
            }
            if any(proof.get(key) != value for key, value in expected_proof.items()):
                raise RepairManifestError(
                    f"Checkpoint mirror proof is invalid: {marker}"
                )
            source_checkpoint = Path(
                str(proof.get("source_checkpoint") or "")
            ).expanduser()
            if (
                not source_checkpoint.is_absolute()
                or source_checkpoint.name != validation.checkpoint_dir.name
            ):
                raise RepairManifestError(
                    f"Checkpoint mirror proof source is invalid: {marker}"
                )
            committed.append(validation)
    if not committed:
        raise RepairManifestError("No fully mirrored Mut generation checkpoint exists")
    selected = committed[-2:]
    checkpoint.mkdir(parents=True, exist_ok=True)
    local_by_step = {
        validate_generation_checkpoint(path).completed_step: path for path in local_valid
    }
    restored: list[int] = []
    for validation in selected:
        existing = local_by_step.get(validation.completed_step)
        if existing is not None:
            observed = validate_generation_checkpoint(existing)
            if observed.checkpoint_digest != validation.checkpoint_digest:
                raise RepairManifestError("Local/mirror checkpoint digest conflict")
            continue
        destination = checkpoint / validation.checkpoint_dir.name
        if destination.exists() or destination.is_symlink():
            raise RepairManifestError(f"Unsafe checkpoint restore destination: {destination}")
        _copy_checkpoint_atomic(validation.checkpoint_dir, destination)
        observed = validate_generation_checkpoint(destination)
        if observed.checkpoint_digest != validation.checkpoint_digest:
            raise RepairManifestError("Restored checkpoint digest mismatch")
        restored.append(validation.completed_step)
    latest = selected[-1]
    _atomic_json(
        checkpoint / LATEST_FILENAME,
        {
            "schema_version": LATEST_SCHEMA_VERSION,
            "completed_step": latest.completed_step,
            "checkpoint_dir": latest.checkpoint_dir.name,
            "checkpoint_digest": latest.checkpoint_digest,
        },
    )
    audit = {
        "schema_version": "mut_traceoff_generation_resume_v1",
        "status": "PASS",
        "selected_step": latest.completed_step,
        "selected_digest": latest.checkpoint_digest,
        "restored_steps": restored,
        "checkpoint_root": str(checkpoint),
        "mirror_root": str(mirror),
        "prepared_at": _utc_now(),
    }
    _atomic_json(output / "resume_preflight.json", audit)
    return audit


def _source_gate_task(
    *, source_root: Path, fresh_root: Path, project_root: Path, proc_root: Path
) -> dict[str, Any]:
    return {
        "id": TRACE_SOURCE_TASK_ID,
        "dataset": "mutagenicity",
        "stage": "MUT_TRACED_SOURCE_AUDIT",
        "runner_dataset": "mut-trace-on-source",
        "runner_stage": "MUT_TRACED_SOURCE_AUDIT",
        "depends_on": [],
        "resource": "cpu",
        "priority": 1,
        "data_splits": [],
        "manifest_only": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/manage_mut_traceoff_parity_v1.py",
            "--config",
            "configs/hpc.yaml",
            "verify-traced-source",
            "--source-root",
            str(source_root),
            "--proc-root",
            str(proc_root),
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(source_root / "run_manifest.json"),
        "config_files": [str(source_root / "resolved_config.json")],
        "expected_output": str(fresh_root / "source-adoptions/trace-on/attempt-{attempt}"),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": "[MUT_TRACE_ON_SOURCE_PASS]",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RUN_TASTEMOLNET": "0",
        },
    }


def _threshold_task(
    *,
    repair_manifest: Path,
    repair_root: Path,
    threshold_output: Path,
    fresh_root: Path,
    project_root: Path,
    control_root: Path,
    proc_root: Path,
) -> dict[str, Any]:
    return {
        "id": THRESHOLD_TASK_ID,
        "dataset": "mutagenicity",
        "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "runner_dataset": "mut-threshold-source",
        "runner_stage": "MUT_THRESHOLD_SOURCE_GATE",
        "depends_on": [],
        "resource": "cpu",
        "priority": 2,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/manage_mut_traceoff_parity_v1.py",
            "--config",
            "configs/hpc.yaml",
            "verify-threshold",
            "--source-manifest",
            str(repair_manifest),
            "--source-controller-root",
            str(repair_root),
            "--control-root",
            str(control_root),
            "--expected-output-root",
            str(threshold_output),
            "--project-root",
            str(project_root),
            "--proc-root",
            str(proc_root),
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(threshold_output / "frozen_threshold_contract.json"),
        "config_files": [str(threshold_output / "threshold_adoption_audit.json")],
        "expected_output": str(fresh_root / "source-adoptions/threshold/attempt-{attempt}"),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": "[MUT_THRESHOLD_SOURCE_PASS]",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "RUN_TASTEMOLNET": "0",
        },
    }


def build_payload(*, spec_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path_value = _absolute(spec_path, label="Mut parity spec", kind="file")
    spec = _read_object(spec_path_value)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RepairManifestError(f"Mut parity spec schema must be {SPEC_SCHEMA}")
    if spec.get("controller_id") != CONTROLLER_ID:
        raise RepairManifestError(f"controller_id must be {CONTROLLER_ID}")
    if spec.get("paper_frozen") is not True or spec.get("run_tastemolnet") != 0:
        raise RepairManifestError("paper_frozen=true and run_tastemolnet=0 are mandatory")
    runtime_root = _absolute(spec.get("runtime_root"), label="runtime root", kind="dir")
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
    if control_root != (runtime_root / "control").resolve(strict=True):
        raise RepairManifestError("control_root must equal runtime_root/control")
    project_root = _absolute(spec.get("project_root"), label="project root", kind="dir")
    python = _absolute(spec.get("python"), label="Python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError("configured Python is not executable")
    proc_root = _absolute(spec.get("proc_root", "/proc"), label="proc root", kind="dir")
    fresh_root = _absolute(spec.get("fresh_output_root"), label="fresh output", kind="fresh")
    expected_fresh_parent = runtime_root / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs"
    try:
        fresh_root.relative_to(expected_fresh_parent)
    except ValueError as exc:
        raise RepairManifestError("fresh output escapes the paper-matrix repairs namespace") from exc
    if fresh_root.exists() or fresh_root.is_symlink():
        raise FileExistsError(f"fresh output already exists: {fresh_root}")
    required_fix = str(spec.get("verify_comrecgc_checkout_safe_git_fix_commit") or "")
    fix = verify_fix_ancestry(project_root=project_root, required_fix_commit=required_fix)
    execution_head = _git_head(project_root)
    if fix.get("execution_head") != execution_head:
        raise RepairManifestError("execution ancestry and HEAD evidence disagree")

    instrumentation = _mapping(
        spec.get("instrumentation_equivalence"),
        label="instrumentation_equivalence",
    )
    legacy_project_root = _absolute(
        instrumentation.get("legacy_project_root"),
        label="legacy 7f scientific worktree",
        kind="dir",
    )
    instrumentation_project_root = _absolute(
        instrumentation.get("instrumentation_project_root"),
        label="checkpoint-instrumented scientific worktree",
        kind="dir",
    )
    if legacy_project_root == instrumentation_project_root:
        raise RepairManifestError("Legacy and instrumentation worktrees must be distinct")
    if _git_head(legacy_project_root) != SOURCE_PROJECT_COMMIT:
        raise RepairManifestError("Legacy scientific worktree is not exact 7f source")
    if _git_head(instrumentation_project_root) != INSTRUMENTATION_PROJECT_COMMIT:
        raise RepairManifestError(
            "Scientific execution worktree is not the reviewed checkpoint instrumentation commit"
        )
    if int(instrumentation.get("steps", -1)) != INSTRUMENTATION_EQUIVALENCE_STEPS:
        raise RepairManifestError("Instrumentation equivalence must use the 500-step prefix")
    if not _git_is_ancestor(
        ancestor=SOURCE_PROJECT_COMMIT,
        descendant=INSTRUMENTATION_PROJECT_COMMIT,
        project_root=instrumentation_project_root,
    ):
        raise RepairManifestError("Checkpoint instrumentation is not a child of 7f")
    _require_clean_tracked_worktree(legacy_project_root, label="legacy scientific")
    _require_clean_tracked_worktree(
        instrumentation_project_root, label="instrumented scientific"
    )
    legacy_source_inventory = instrumentation_source_inventory(legacy_project_root)
    instrumentation_source_inventory_value = instrumentation_source_inventory(
        instrumentation_project_root
    )
    if (
        legacy_source_inventory.get("inventory_sha256")
        != LEGACY_SOURCE_INVENTORY_SHA256
        or instrumentation_source_inventory_value.get("inventory_sha256")
        != INSTRUMENTATION_SOURCE_INVENTORY_SHA256
    ):
        raise RepairManifestError(
            "Scientific source inventory differs from the reviewed 7f/664 pair"
        )

    source_root = _absolute(spec.get("traced_source_root"), label="traced source", kind="dir")
    source_evidence = verify_traced_source(
        source_root=source_root, proc_root=proc_root, hash_payload=False
    )

    replay = _mapping(spec.get("replay"), label="replay")
    upstream = _absolute(replay.get("upstream_root"), label="upstream root", kind="dir")
    upstream_evidence = verify_checkout(
        upstream, expected_commit=SOURCE_UPSTREAM_COMMIT, validate_imports=False
    )
    dataset_dir = _absolute(replay.get("dataset_dir"), label="dataset dir", kind="dir")
    gnn = _absolute(replay.get("gnn_checkpoint"), label="GNN checkpoint", kind="file")
    distance = _absolute(
        replay.get("distance_checkpoint"), label="distance checkpoint", kind="file"
    )
    replay_failures: list[str] = []
    if sha256_file(gnn) != SOURCE_GNN_SHA256:
        replay_failures.append("gnn_checkpoint_sha256")
    if sha256_file(distance) != SOURCE_DISTANCE_SHA256:
        replay_failures.append("distance_checkpoint_sha256")
    if int(replay.get("parent_limit", -1)) != SOURCE_PARENT_COUNT:
        replay_failures.append("parent_limit")
    if int(replay.get("batch_size", -1)) != SOURCE_BATCH_SIZE:
        replay_failures.append("batch_size")
    if replay.get("parameters") != SOURCE_PARAMETERS:
        replay_failures.append("parameters")
    if replay.get("trace_enabled") is not False:
        replay_failures.append("trace_enabled")
    if replay_failures:
        raise RepairManifestError(f"Mut replay differs from frozen science: {replay_failures}")

    highmem_lock = _absolute(
        spec.get("highmem_lock_path"), label="highmem lock", kind="fresh"
    )
    expected_lock = (runtime_root / "locks/comrecgc_common_recourse_highmem.lock").resolve(
        strict=False
    )
    if highmem_lock != expected_lock:
        raise RepairManifestError(f"highmem lock must be exact: {expected_lock}")
    flock_bin = _absolute(spec.get("flock_bin"), label="flock", kind="file")
    cgroup = _absolute(spec.get("cgroup_memory_root"), label="cgroup root", kind="dir")
    min_free_bytes = int(spec.get("min_cgroup_free_bytes", 0))
    if min_free_bytes < AIDS_MINIMUM_HEADROOM_BYTES:
        raise RepairManifestError(
            "Mut replay requires the shared 400 GiB cgroup headroom contract"
        )

    repair_v2 = _mapping(spec.get("repair_v2"), label="repair_v2")
    repair_v2_manifest = _absolute(
        repair_v2.get("manifest"), label="repair-v2 manifest", kind="file"
    )
    repair_v2_root = _absolute(
        repair_v2.get("root"), label="repair-v2 controller root", kind="dir"
    )
    failed_mut = _absolute(
        repair_v2.get("failed_mut_output"), label="repair-v2 Mut output", kind="dir"
    )
    repair_v2_controller_evidence = _repair_v2_manifest(
        source_manifest=repair_v2_manifest,
        source_controller_root=repair_v2_root,
        control_root=control_root,
        expected_output_root=failed_mut,
    )
    common_evidence = _validate_common_recourse_source(
        repair_v2_output=failed_mut, proc_root=proc_root
    )

    repair_v1 = _mapping(spec.get("repair_v1"), label="repair_v1")
    repair_v1_manifest = _absolute(
        repair_v1.get("manifest"), label="repair-v1 manifest", kind="file"
    )
    repair_v1_root = _absolute(
        repair_v1.get("root"), label="repair-v1 controller root", kind="dir"
    )
    threshold_output = _absolute(
        repair_v1.get("mut_threshold_output"), label="Mut threshold output", kind="dir"
    )
    threshold_evidence = verify_repair_v1_source(
        source_key="mut_threshold",
        source_manifest=repair_v1_manifest,
        source_controller_root=repair_v1_root,
        control_root=control_root,
        expected_output_root=threshold_output,
        project_root=project_root,
        required_fix_commit=required_fix,
        proc_root=proc_root,
    )
    threshold_contract = _absolute(
        _mapping(threshold_evidence.get("semantic"), label="threshold semantic").get(
            "threshold_contract"
        ),
        label="threshold contract",
        kind="file",
    )

    aids = _mapping(spec.get("aids_dependency"), label="aids_dependency")
    aids_controller_id = str(aids.get("controller_id") or "")
    aids_task_id = str(aids.get("task_id") or "")
    aids_wrapper = str(aids.get("wrapper") or "")
    aids_terminal_contract = str(aids.get("terminal_contract") or "")
    if aids_terminal_contract != "comrecgc_standardized_v1":
        raise RepairManifestError("AIDS dependency terminal contract is unsupported")
    aids_manifest_sha256 = str(aids.get("expected_manifest_sha256") or "")
    if HEX64.fullmatch(aids_manifest_sha256) is None:
        raise RepairManifestError("AIDS dependency expected manifest SHA256 is malformed")
    aids_min_free_bytes = int(aids.get("min_cgroup_free_bytes", 0))
    if aids_min_free_bytes != AIDS_V4_MINIMUM_HEADROOM_BYTES:
        raise RepairManifestError(
            "AIDS-v4 dependency must freeze its reviewed 128 GiB headroom"
        )
    aids_manifest = _absolute(
        aids.get("manifest"), label="AIDS dependency manifest", kind="file"
    )
    aids_root = _absolute(
        aids.get("root"), label="AIDS dependency root", kind="dir"
    )
    aids_output = _absolute(
        aids.get("expected_output"), label="AIDS dependency expected output", kind="fresh"
    )
    aids_controller = _aids_manifest(
        source_manifest=aids_manifest,
        source_controller_root=aids_root,
        control_root=control_root,
        expected_controller_id=aids_controller_id,
        expected_task_id=aids_task_id,
        expected_wrapper=aids_wrapper,
        expected_manifest_sha256=aids_manifest_sha256,
        expected_highmem_lock=highmem_lock,
        expected_flock_bin=flock_bin,
        expected_cgroup_root=cgroup,
        expected_min_free_bytes=aids_min_free_bytes,
        expected_proc_root=proc_root,
    )

    science = _mapping(spec.get("standardization"), label="standardization")
    directories = ("molclr_root",)
    files = ("dataset_csv", "teacher_path", "molclr_checkpoint")
    standardized_paths: dict[str, Path] = {}
    for key in directories:
        standardized_paths[key] = _absolute(science.get(key), label=key, kind="dir")
    for key in files:
        standardized_paths[key] = _absolute(science.get(key), label=key, kind="file")

    wrapper = "{project_root}/scripts/autodl/run_mut_traceoff_stage_highmem.sh"
    common_root = Path(str(common_evidence["source_common_recourse_root"]))
    equivalence_run_root_template = str(
        fresh_root / "checkpoint-instrumentation-equivalence-runs/attempt-{attempt}"
    )
    equivalence_gate_template = str(
        fresh_root / "checkpoint-instrumentation-equivalence/attempt-{attempt}"
    )
    generation_root = fresh_root / "traceoff-generation"
    checkpoint_root = generation_root / "generation_checkpoints"
    mirror_root = fresh_root / "traceoff-generation-checkpoint-mirror"
    traceoff_scientific_argv = expected_traceoff_scientific_argv(
        instrumentation_project_root=instrumentation_project_root,
        upstream_root=upstream,
        dataset_dir=dataset_dir,
        gnn_checkpoint=gnn,
        distance_checkpoint=distance,
        generation_root=generation_root,
        checkpoint_root=checkpoint_root,
        mirror_root=mirror_root,
    )
    traceoff_scientific_command_sha256 = scientific_command_sha256(
        traceoff_scientific_argv
    )
    generation_gate_template = str(
        fresh_root / "traceoff-reference-gates/attempt-{attempt}"
    )
    parity_template = str(fresh_root / "trace-parity/attempt-{attempt}")
    common_gate_template = str(fresh_root / "common-adoption/attempt-{attempt}")
    standardized_template = str(
        fresh_root
        / "cells/mutagenicity/comrecgc/standardized/attempt-{attempt}"
    )
    base_env = {
        "AUTODL_PYTHON": "{python}",
        "MUT_CONTROLLER_PROJECT_ROOT": "{project_root}",
        "MUT_INSTRUMENTATION_PROJECT_ROOT": str(instrumentation_project_root),
        "MUT_LEGACY_PROJECT_ROOT": str(legacy_project_root),
        "PYTHONPATH": "{project_root}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "RUN_TASTEMOLNET": "0",
        "COMRECGC_HIGHMEM_LOCK_PATH": str(highmem_lock),
        "COMRECGC_FLOCK_BIN": str(flock_bin),
        "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup),
        "COMRECGC_MIN_CGROUP_FREE_BYTES": str(min_free_bytes),
        "COMRECGC_PROC_ROOT": str(proc_root),
        "MUT_CONTROLLER_COMMIT": execution_head,
        "MUT_EXECUTION_COMMIT": INSTRUMENTATION_PROJECT_COMMIT,
        "MUT_SOURCE_PROJECT_COMMIT": SOURCE_PROJECT_COMMIT,
        "MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256": (
            traceoff_scientific_command_sha256
        ),
        "MUT_LEGACY_SOURCE_INVENTORY_SHA256": str(
            legacy_source_inventory["inventory_sha256"]
        ),
        "MUT_INSTRUMENTATION_SOURCE_INVENTORY_SHA256": str(
            instrumentation_source_inventory_value["inventory_sha256"]
        ),
    }
    tasks: list[dict[str, Any]] = [
        _source_gate_task(
            source_root=source_root,
            fresh_root=fresh_root,
            project_root=project_root,
            proc_root=proc_root,
        ),
        _threshold_task(
            repair_manifest=repair_v1_manifest,
            repair_root=repair_v1_root,
            threshold_output=threshold_output,
            fresh_root=fresh_root,
            project_root=project_root,
            control_root=control_root,
            proc_root=proc_root,
        ),
        {
            "id": AIDS_WAIT_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "MUT_WAIT_AIDS_COMRECGC_PASS",
            "runner_dataset": "mut-wait-aids-comrecgc",
            "runner_stage": "MUT_WAIT_AIDS_COMRECGC_PASS",
            "depends_on": [],
            "resource": "cpu",
            "priority": 3,
            "data_splits": [],
            "manifest_only": True,
            "command": [
                "{python}",
                "{project_root}/scripts/autodl/manage_mut_traceoff_parity_v1.py",
                "--config",
                "configs/hpc.yaml",
                "wait-aids",
                "--expected-controller-id",
                aids_controller_id,
                "--expected-task-id",
                aids_task_id,
                "--expected-wrapper",
                aids_wrapper,
                "--expected-manifest-sha256",
                aids_manifest_sha256,
                "--source-manifest",
                str(aids_manifest),
                "--source-controller-root",
                str(aids_root),
                "--control-root",
                str(control_root),
                "--expected-output-root",
                str(aids_output),
                "--proc-root",
                str(proc_root),
                "--poll-seconds",
                "60",
                "--output-dir",
                "{task_output}",
            ],
            "input_manifest": str(aids_manifest),
            "config_files": [str(aids_root / "controller_manifest.json")],
            "expected_output": str(
                fresh_root / "dependencies/aids-comrecgc/attempt-{attempt}"
            ),
            "required_output_files": ["aids_dependency.json", "PASS"],
            "required_log_marker": "[MUT_AIDS_DEPENDENCY_PASS]",
            "environment": {
                "PYTHONPATH": "{project_root}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "RUN_TASTEMOLNET": "0",
            },
        },
        {
            "id": INSTRUMENTATION_EQUIVALENCE_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE",
            "runner_dataset": "mut-checkpoint-instrumentation-equivalence",
            "runner_stage": "MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE",
            "depends_on": [TRACE_SOURCE_TASK_ID, AIDS_WAIT_TASK_ID],
            "resource": "gpu",
            "gpu_lock_mode": "exclusive",
            "priority": 8,
            "data_splits": ["generation_source"],
            "manifest_only": False,
            "command": ["bash", wrapper],
            "input_manifest": "{dep_"
            + TRACE_SOURCE_TASK_ID
            + "_output}/source_gate.json",
            "config_files": [str(source_root / "resolved_config.json")],
            "expected_output": equivalence_gate_template,
            "required_output_files": ["equivalence.json", "PASS"],
            "required_log_marker": (
                "[MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE_PASS]"
            ),
            "environment": {
                **base_env,
                "MUT_TRACEOFF_STAGE": "instrumentation-equivalence",
                "GPU_REQUIRED": "1",
                "DEVICE": "cuda:0",
                "MUT_SOURCE_ROOT": str(source_root),
                "MUT_UPSTREAM_ROOT": str(upstream),
                "MUT_DATASET_DIR": str(dataset_dir),
                "MUT_GNN_CHECKPOINT": str(gnn),
                "MUT_DISTANCE_CHECKPOINT": str(distance),
                "MUT_EQUIVALENCE_RUN_ROOT": equivalence_run_root_template,
                "MUT_BATCH_SIZE": str(SOURCE_BATCH_SIZE),
                "MUT_STAGE_OUTPUT": "{task_output}",
            },
            "semantic_failure_markers": [
                "legacy scientific worktree changed",
                "instrumentation execution commit changed",
                "step_action_trace",
                "rng_state",
                "payload",
                "checkpoint",
                "test leakage",
            ],
        },
        {
            "id": TRACEOFF_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "MUT_TRACEOFF_REFERENCE_50K",
            "runner_dataset": "mut-traceoff-reference",
            "runner_stage": "MUT_TRACEOFF_REFERENCE_50K",
            "depends_on": [
                TRACE_SOURCE_TASK_ID,
                AIDS_WAIT_TASK_ID,
                INSTRUMENTATION_EQUIVALENCE_TASK_ID,
            ],
            "resource": "gpu",
            "gpu_lock_mode": "exclusive",
            "priority": 10,
            "data_splits": ["generation_source"],
            "manifest_only": False,
            "command": ["bash", wrapper],
            "input_manifest": "{dep_" + TRACE_SOURCE_TASK_ID + "_output}/source_gate.json",
            "config_files": [str(source_root / "resolved_config.json")],
            "expected_output": generation_gate_template,
            "required_output_files": [
                "traceoff_reference.json",
                "PASS",
            ],
            "required_log_marker": "[MUT_TRACEOFF_REFERENCE_PASS]",
            "environment": {
                **base_env,
                "MUT_TRACEOFF_STAGE": "generation",
                "GPU_REQUIRED": "1",
                "DEVICE": "cuda:0",
                "MUT_SOURCE_ROOT": str(source_root),
                "MUT_UPSTREAM_ROOT": str(upstream),
                "MUT_DATASET_DIR": str(dataset_dir),
                "MUT_GNN_CHECKPOINT": str(gnn),
                "MUT_DISTANCE_CHECKPOINT": str(distance),
                "MUT_GENERATION_OUTPUT": str(generation_root),
                "MUT_CHECKPOINT_ROOT": str(checkpoint_root),
                "MUT_CHECKPOINT_MIRROR_ROOT": str(mirror_root),
                "MUT_BATCH_SIZE": str(SOURCE_BATCH_SIZE),
                "MUT_INSTRUMENTATION_EQUIVALENCE_GATE": "{dep_"
                + INSTRUMENTATION_EQUIVALENCE_TASK_ID
                + "_output}/equivalence.json",
                "MUT_STAGE_OUTPUT": "{task_output}",
            },
            "semantic_failure_markers": [
                "source closure changed",
                "checkpoint digest mismatch",
                "scientific argv",
                "selected comrecgc transition is not one unique",
                "test leakage",
            ],
        },
        {
            "id": PARITY_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "MUT_ASSERT_TRACE_PARITY",
            "runner_dataset": "mut-trace-parity",
            "runner_stage": "MUT_ASSERT_TRACE_PARITY",
            "depends_on": [TRACE_SOURCE_TASK_ID, TRACEOFF_TASK_ID],
            "resource": "cpu",
            "priority": 20,
            "data_splits": [],
            "manifest_only": True,
            "command": ["bash", wrapper],
            "input_manifest": "{dep_" + TRACEOFF_TASK_ID + "_output}/traceoff_reference.json",
            "config_files": [str(source_root / "run_manifest.json")],
            "expected_output": parity_template,
            "required_output_files": ["trace_parity.json", "PASS"],
            "required_log_marker": "[MUT_TRACE_PARITY_PASS]",
            "environment": {
                **base_env,
                "MUT_TRACEOFF_STAGE": "parity",
                "GPU_REQUIRED": "0",
                "DEVICE": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "MUT_SOURCE_ROOT": str(source_root),
                "MUT_GENERATION_OUTPUT": str(generation_root),
                "MUT_STAGE_OUTPUT": "{task_output}",
            },
        },
        {
            "id": COMMON_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "MUT_ADOPT_COMMON_RECOURSE",
            "runner_dataset": "mut-common-adoption",
            "runner_stage": "MUT_ADOPT_COMMON_RECOURSE",
            "depends_on": [PARITY_TASK_ID],
            "resource": "cpu",
            "priority": 30,
            "data_splits": [],
            "manifest_only": True,
            "command": [
                "{python}",
                "{project_root}/scripts/autodl/manage_mut_traceoff_parity_v1.py",
                "--config",
                "configs/hpc.yaml",
                "adopt-common",
                "--repair-v2-output",
                str(failed_mut),
                "--parity-gate",
                "{dep_" + PARITY_TASK_ID + "_output}/trace_parity.json",
                "--proc-root",
                str(proc_root),
                "--output-dir",
                "{task_output}",
            ],
            "input_manifest": "{dep_" + PARITY_TASK_ID + "_output}/trace_parity.json",
            "config_files": [str(common_root / "run_manifest.json")],
            "expected_output": common_gate_template,
            "required_output_files": ["common_recourse_adoption.json", "PASS"],
            "required_log_marker": "[MUT_COMMON_ADOPTION_PASS]",
            "environment": {
                "PYTHONPATH": "{project_root}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "RUN_TASTEMOLNET": "0",
            },
        },
        {
            "id": STANDARDIZE_TASK_ID,
            "dataset": "mutagenicity",
            "stage": "AM_COMRECGC_HELDOUT_EVAL",
            "runner_dataset": "paper-cell-mutagenicity-comrecgc-traceoff-v1",
            "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
            "depends_on": [THRESHOLD_TASK_ID, PARITY_TASK_ID, COMMON_TASK_ID],
            "resource": "cpu",
            "priority": 40,
            "data_splits": ["test"],
            "manifest_only": False,
            "selector_parameters_frozen": True,
            "read_only_test": True,
            "command": ["bash", wrapper],
            "input_manifest": "{dep_" + COMMON_TASK_ID + "_output}/common_recourse_adoption.json",
            "config_files": [str(threshold_contract)],
            "expected_output": standardized_template,
            "required_output_files": [
                "generation_adoption_manifest.json",
                "common_recourse_adoption_manifest.json",
                "trace_parity_adoption_manifest.json",
                "standardized/_FINALIZED.json",
                "standardized/run_manifest.json",
                "standardized/adoption_manifest.json",
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "PASS",
            ],
            "required_log_marker": "[MUT_COMRECGC_PARITY_STANDARDIZATION_PASS]",
            "environment": {
                **base_env,
                "MUT_TRACEOFF_STAGE": "standardization",
                "GPU_REQUIRED": "0",
                "DEVICE": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "MUT_SOURCE_ROOT": str(source_root),
                "MUT_UPSTREAM_ROOT": str(upstream),
                "MUT_DATASET_DIR": str(dataset_dir),
                "MUT_DISTANCE_CHECKPOINT": str(distance),
                "MUT_DATASET_CSV": str(standardized_paths["dataset_csv"]),
                "MUT_TEACHER_PATH": str(standardized_paths["teacher_path"]),
                "MUT_MOLCLR_ROOT": str(standardized_paths["molclr_root"]),
                "MUT_MOLCLR_CHECKPOINT": str(standardized_paths["molclr_checkpoint"]),
                "MUT_THRESHOLDS_PATH": str(threshold_contract),
                "MUT_COMMON_RECOURSE_ROOT": str(common_root),
                "MUT_COMMON_ADOPTION_GATE": "{dep_"
                + COMMON_TASK_ID
                + "_output}/common_recourse_adoption.json",
                "MUT_PARITY_GATE": "{dep_"
                + PARITY_TASK_ID
                + "_output}/trace_parity.json",
                "MUT_STAGE_OUTPUT": "{task_output}",
            },
            "semantic_failure_markers": [
                "trace parity gate is invalid",
                "chemistry repair cannot be frozen",
                "source closure changed",
                "live writer detected",
                "test leakage",
            ],
        },
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "controller_id": CONTROLLER_ID,
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "min_free_memory_mb": 16_000,
            "idle_util_threshold": 10,
            "worker_launcher": "auto",
            "max_cpu_tasks": 1,
            "launch_grace_seconds": 180,
            "max_transient_retries": 1,
            "keep_alive_when_blocked": True,
        },
        "resource_gates": {
            "min_available_ram_gb": 64,
            "min_free_disk_gb": 100,
            "max_cpu_load_fraction": 0.9,
        },
        "mut_traceoff_parity_contract": {
            "schema_version": SPEC_SCHEMA,
            "spec_path": str(spec_path_value),
            "spec_sha256": sha256_file(spec_path_value),
            "controller_project_root": str(project_root),
            "controller_commit": execution_head,
            "source_algorithm_project_root": str(legacy_project_root),
            "source_algorithm_commit": SOURCE_PROJECT_COMMIT,
            "execution_instrumentation_project_root": str(
                instrumentation_project_root
            ),
            "execution_instrumentation_commit": INSTRUMENTATION_PROJECT_COMMIT,
            # Compatibility aliases remain explicit and truthful: execution is
            # the instrumentation commit, never the historical 7f source.
            "execution_project_root": str(instrumentation_project_root),
            "execution_commit": INSTRUMENTATION_PROJECT_COMMIT,
            "source_project_commit": SOURCE_PROJECT_COMMIT,
            "instrumentation_intended_effect": (
                "completed_step_checkpoint_and_exact_resume_only_for_mut_generation"
            ),
            "instrumentation_source_text_identical": False,
            "instrumentation_behavioral_equivalence_required": True,
            "instrumentation_equivalence_steps": INSTRUMENTATION_EQUIVALENCE_STEPS,
            "instrumentation_equivalence_test_split_loaded": False,
            "instrumentation_equivalence_calibration_split_loaded": False,
            "legacy_source_inventory": legacy_source_inventory,
            "instrumentation_source_inventory": (
                instrumentation_source_inventory_value
            ),
            "instrumentation_equivalence_run_root_template": (
                equivalence_run_root_template
            ),
            "instrumentation_equivalence_output_template": (
                equivalence_gate_template
            ),
            "traceoff_scientific_argv": list(traceoff_scientific_argv),
            "traceoff_scientific_command_sha256": (
                traceoff_scientific_command_sha256
            ),
            "source_upstream_commit": SOURCE_UPSTREAM_COMMIT,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_dataset_sha256": SOURCE_DATASET_SHA256,
            "source_parent_order_sha256": SOURCE_PARENT_ORDER_SHA256,
            "source_parameters": dict(SOURCE_PARAMETERS),
            "upstream_evidence": upstream_evidence,
            "source_trace_enabled": True,
            "reference_trace_enabled": False,
            "reference_fresh": True,
            "reference_self_comparison_forbidden": True,
            "trace_fields_stripped": False,
            "generation_gpu_required": True,
            "generation_gpu_lock_mode": "exclusive",
            "standardization_gpu_required": False,
            "waits_for_reviewed_aids_repair_pass": True,
            "highmem_lock_path": str(highmem_lock),
            "source_evidence": source_evidence,
            "threshold_evidence": threshold_evidence,
            "common_recourse_evidence": common_evidence,
            "repair_v2_controller_evidence": repair_v2_controller_evidence,
            "aids_controller_id": aids_controller.controller_id,
            "aids_task_id": aids_task_id,
            "aids_wrapper": aids_wrapper,
            "aids_terminal_contract": aids_terminal_contract,
            "aids_expected_manifest_sha256": aids_manifest_sha256,
            "aids_manifest_sha256": aids_controller.sha256,
            "aids_min_cgroup_free_bytes": aids_min_free_bytes,
            "mut_min_cgroup_free_bytes": min_free_bytes,
            "fresh_output_root": str(fresh_root),
            "traceoff_output_root": str(generation_root),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_mirror_root": str(mirror_root),
            "traceoff_gate_output_template": generation_gate_template,
            "parity_output_template": parity_template,
            "common_adoption_output_template": common_gate_template,
            "standardized_output_template": standardized_template,
            "taste_tasks_present": False,
            "bace_tasks_present": False,
            "paper_tasks_present": False,
        },
        "tasks": tasks,
    }
    validation = validate_payload(payload)
    return payload, {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_count": validation["task_count"],
        "fresh_output_root": str(fresh_root),
        "generation_resource": "exclusive_gpu",
        "standardization_resource": "cpu",
        "estimated_traceoff_hours": "73-96",
    }


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mut-traceoff-parity-v1-") as directory:
        manifest_path = Path(directory) / "manifest.json"
        _atomic_json(manifest_path, payload)
        manifest = load_controller_manifest(manifest_path)
    expected = {
        TRACE_SOURCE_TASK_ID,
        THRESHOLD_TASK_ID,
        AIDS_WAIT_TASK_ID,
        INSTRUMENTATION_EQUIVALENCE_TASK_ID,
        TRACEOFF_TASK_ID,
        PARITY_TASK_ID,
        COMMON_TASK_ID,
        STANDARDIZE_TASK_ID,
    }
    failures: list[str] = []
    runtime = _mapping(payload.get("runtime"), label="runtime")
    if int(runtime.get("max_cpu_tasks", -1)) != 1:
        failures.append("max_cpu_tasks")
    if manifest.controller_id != CONTROLLER_ID:
        failures.append("controller_id")
    if {task.task_id for task in manifest.tasks} != expected or len(manifest.tasks) != 8:
        failures.append("task_boundary")
    equivalence = manifest.by_id.get(INSTRUMENTATION_EQUIVALENCE_TASK_ID)
    traceoff = manifest.by_id.get(TRACEOFF_TASK_ID)
    parity = manifest.by_id.get(PARITY_TASK_ID)
    standard = manifest.by_id.get(STANDARDIZE_TASK_ID)
    if (
        equivalence is None
        or equivalence.resource != "gpu"
        or equivalence.gpu_lock_mode != "exclusive"
        or equivalence.environment.get("MUT_TRACEOFF_STAGE")
        != "instrumentation-equivalence"
        or equivalence.environment.get("GPU_REQUIRED") != "1"
        or equivalence.data_splits != ("generation_source",)
        or AIDS_WAIT_TASK_ID not in equivalence.depends_on
    ):
        failures.append("instrumentation_equivalence_task")
    if traceoff is None or traceoff.resource != "gpu" or traceoff.gpu_lock_mode != "exclusive":
        failures.append("traceoff_resource")
    if traceoff is not None and traceoff.environment.get("MUT_TRACEOFF_STAGE") != "generation":
        failures.append("traceoff_stage")
    if traceoff is not None and "MUT_TRACE_OUTPUT" in traceoff.environment:
        failures.append("trace_output_present")
    if (
        traceoff is not None
        and INSTRUMENTATION_EQUIVALENCE_TASK_ID not in traceoff.depends_on
    ):
        failures.append("traceoff_equivalence_dependency")
    if parity is None or parity.resource != "cpu" or parity.environment.get("DEVICE") != "cpu":
        failures.append("parity_resource")
    if standard is None or standard.resource != "cpu":
        failures.append("standardization_resource")
    if standard is not None and (
        standard.environment.get("GPU_REQUIRED") != "0"
        or standard.environment.get("CUDA_VISIBLE_DEVICES") != ""
        or standard.data_splits != ("test",)
        or not standard.selector_parameters_frozen
        or not standard.read_only_test
    ):
        failures.append("standardization_contract")
    if standard is not None and COMMON_TASK_ID not in standard.depends_on:
        failures.append("common_dependency")
    contract = _mapping(payload.get("mut_traceoff_parity_contract"), label="contract")
    contract_scientific_argv = tuple(
        str(value) for value in contract.get("traceoff_scientific_argv") or ()
    )
    if (
        not contract_scientific_argv
        or scientific_command_sha256(contract_scientific_argv)
        != contract.get("traceoff_scientific_command_sha256")
        or traceoff is None
        or traceoff.environment.get("MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256")
        != contract.get("traceoff_scientific_command_sha256")
    ):
        failures.append("scientific_argv_contract")
    if (
        contract.get("source_project_commit") != SOURCE_PROJECT_COMMIT
        or contract.get("source_algorithm_commit") != SOURCE_PROJECT_COMMIT
        or contract.get("execution_instrumentation_commit")
        != INSTRUMENTATION_PROJECT_COMMIT
        or contract.get("execution_commit") != INSTRUMENTATION_PROJECT_COMMIT
        or contract.get("instrumentation_behavioral_equivalence_required") is not True
        or int(contract.get("instrumentation_equivalence_steps", -1))
        != INSTRUMENTATION_EQUIVALENCE_STEPS
        or contract.get("source_trace_enabled") is not True
        or contract.get("reference_trace_enabled") is not False
        or contract.get("reference_self_comparison_forbidden") is not True
        or contract.get("trace_fields_stripped") is not False
        or contract.get("waits_for_reviewed_aids_repair_pass") is not True
        or contract.get("aids_controller_id") == AIDS_CONTROLLER_ID
        or int(contract.get("aids_min_cgroup_free_bytes", -1))
        != AIDS_V4_MINIMUM_HEADROOM_BYTES
        or int(contract.get("mut_min_cgroup_free_bytes", -1))
        < AIDS_MINIMUM_HEADROOM_BYTES
        or contract.get("aids_expected_manifest_sha256")
        != contract.get("aids_manifest_sha256")
    ):
        failures.append("scientific_contract")
    if failures:
        raise RepairManifestError(f"Mut trace-off controller is invalid: {failures}")
    return {
        "status": "PASS",
        "controller_id": manifest.controller_id,
        "task_count": len(manifest.tasks),
        "task_ids": [task.task_id for task in manifest.tasks],
        "manifest_sha256": manifest.sha256,
    }


def build_manifest(*, spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    destination = _absolute(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"manifest output must be fresh: {destination}")
    payload, summary = build_payload(spec_path=spec_path)
    spec = _read_object(spec_path)
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
    expected = (
        control_root / SOURCE_NAMESPACE / "manifests" / f"{CONTROLLER_ID}.json"
    ).resolve(strict=False)
    if destination != expected:
        raise RepairManifestError(f"manifest output must be exact: {expected}")
    validation = validate_payload(payload)
    _atomic_json(destination, payload)
    frozen = load_controller_manifest(destination)
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("published manifest differs from validated bytes")
    return {**summary, "manifest": str(destination), "manifest_sha256": frozen.sha256}


__all__ = [
    "AIDS_CONTROLLER_ID",
    "AIDS_V4_MINIMUM_HEADROOM_BYTES",
    "AIDS_TASK_ID",
    "CONTROLLER_ID",
    "INSTRUMENTATION_EQUIVALENCE_STEPS",
    "INSTRUMENTATION_EQUIVALENCE_TASK_ID",
    "INSTRUMENTATION_PROJECT_COMMIT",
    "INSTRUMENTATION_SOURCE_INVENTORY_SHA256",
    "LEGACY_SOURCE_INVENTORY_SHA256",
    "SOURCE_CANDIDATE_COUNT",
    "SOURCE_CONFIG_SHA256",
    "SOURCE_DATASET_SHA256",
    "SOURCE_DISTANCE_SHA256",
    "SOURCE_GNN_SHA256",
    "SOURCE_PARENT_COUNT",
    "SOURCE_PARENT_ORDER_SHA256",
    "SOURCE_PARAMETERS",
    "SOURCE_PAYLOAD_SHA256",
    "SOURCE_PROJECT_COMMIT",
    "SOURCE_STEPS",
    "SPEC_SCHEMA",
    "assert_mut_trace_parity",
    "build_manifest",
    "build_payload",
    "expected_traceoff_scientific_argv",
    "instrumentation_source_inventory",
    "prepare_generation_resume",
    "publish_common_adoption_gate",
    "publish_threshold_source_gate",
    "publish_traced_source_gate",
    "validate_payload",
    "validate_instrumentation_equivalence_gate",
    "verify_traced_source",
    "wait_for_aids_pass",
]
