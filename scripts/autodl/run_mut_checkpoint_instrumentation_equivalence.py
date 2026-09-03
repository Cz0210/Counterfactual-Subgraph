#!/usr/bin/env python3
"""Run a fresh 500-step legacy-vs-checkpointed Mut ComRecGC equivalence gate.

The legacy child imports the exact 7f7ed51 worktree.  The instrumented child
imports the immutable continuation worktree and enables completed-step
checkpointing.  Both use the same full-budget parameters except for the
preregistered 500-step diagnostic prefix and emit an action trace only for the
equivalence audit.  Neither diagnostic output is paper eligible.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


SOURCE_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
INSTRUMENTATION_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
LEGACY_SOURCE_INVENTORY_SHA256 = (
    "240db0f3bfe6c02ef7e60798d7e6ae40c9494d2aae8befe5f687bdda4324c390"
)
INSTRUMENTATION_SOURCE_INVENTORY_SHA256 = (
    "6b3f509ff01059e54006053981c1f8914eacba2bbfd42c3787f9566c626ff1c6"
)
SEMANTIC_FINALIZER_COMMIT = "582bc4b44fbc8ab4595a35da78e4f595e0c0dc5b"
SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256 = (
    "8933af21d007cbccdb12e01fa1634d4668aba1125d63f54971db31c9d6f36a67"
)
SEMANTIC_FINALIZER_SCHEMA = "mut_semantic_lineage_finalizer_receipt_v1"
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
STEPS = 500
SCHEMA = "mut_checkpoint_instrumentation_equivalence_v1"
SOURCE_FILES = (
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


def _absolute(value: str, *, exists: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"physical absolute path required: {value}")
    return path.resolve(strict=exists)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if len(value) != 40:
        raise ValueError(f"Malformed Git HEAD: {value!r}")
    return value


def _git_science_status(root: Path) -> str:
    """Return tracked-worktree drift without treating diagnostic files as code.

    The reviewed 582 worktree may contain unrelated untracked controller
    receipts after deployment.  Those cannot shadow an explicitly loaded
    physical ``graph_trace.py`` and therefore are not a reason to invalidate
    the binding.  Any tracked edit is fatal.
    """

    try:
        return subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Cannot verify semantic finalizer worktree status") from exc


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _rng_payload(torch: Any, np: Any) -> dict[str, Any]:
    return {
        "python": _plain(random.getstate()),
        "numpy": _plain(np.random.get_state()),
        "torch_cpu": _plain(torch.get_rng_state()),
        "torch_cuda": (
            [_plain(value) for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _install_science_root(science_root: Path) -> None:
    retained = []
    for value in sys.path:
        try:
            resolved = Path(value).resolve()
        except (OSError, ValueError):
            retained.append(value)
            continue
        if (resolved / "src").is_dir() or resolved.name == "autodl":
            continue
        retained.append(value)
    sys.path[:] = [str(science_root), *retained]


def _load_semantic_finalizer(
    project_root: Path,
) -> tuple[ModuleType, dict[str, Any]]:
    """Load the reviewed 582 resolver without shadowing frozen science code.

    The 500-step generation continues to import all scientific modules from
    the exact 7f/664 worktree.  Only the post-walk lineage iterator is loaded
    under a private package name from the immutable 582 worktree.  Loading it
    under ``src`` would poison Python's module cache and could silently route
    the random walk through the controller checkout, so that is forbidden.
    """

    root = _absolute(project_root)
    if _git_head(root) != SEMANTIC_FINALIZER_COMMIT:
        raise ValueError("Semantic finalizer worktree commit changed")
    if _git_science_status(root):
        raise ValueError("Semantic finalizer worktree has dirty/shadow source")
    source = root / "src/baselines/comrecgc/graph_trace.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Semantic finalizer graph_trace.py is absent or indirect")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256:
        raise ValueError("Semantic finalizer graph_trace.py changed")

    package_name = "_mut_semantic_finalizer_582bc4b"
    package_root = source.parent
    if package_name in sys.modules:
        raise ValueError("Semantic finalizer private package was already imported")
    package = ModuleType(package_name)
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    module_name = f"{package_name}.graph_trace"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ValueError("Cannot load semantic finalizer graph_trace.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if loaded_source != source.resolve(strict=True):
        raise ValueError("Semantic finalizer module escaped its immutable worktree")
    for symbol in (
        "SemanticTransitionKey",
        "collapse_semantic_transition_aliases",
        "iter_candidate_lineage_from_selected_trace",
    ):
        if not callable(getattr(module, symbol, None)):
            raise ValueError(f"Semantic finalizer lacks reviewed symbol: {symbol}")
    return module, {
        "schema_version": "mut_semantic_lineage_finalizer_binding_v1",
        "status": "PASS",
        "project_root": str(root),
        "commit": SEMANTIC_FINALIZER_COMMIT,
        "graph_trace_path": str(source.resolve(strict=True)),
        "graph_trace_sha256": source_sha256,
        "private_module_name": module_name,
        "resolver_symbol": "collapse_semantic_transition_aliases",
        "lineage_symbol": "iter_candidate_lineage_from_selected_trace",
        "generation_module_namespace": "src.baselines.comrecgc",
        "finalizer_module_namespace": package_name,
        "module_namespaces_disjoint": True,
    }


def _install_semantic_lineage_finalizer(
    *,
    science_graph_trace: ModuleType,
    semantic_graph_trace: ModuleType,
    binding: Mapping[str, Any],
    receipt_path: Path,
    role: str,
    science_project_commit: str,
) -> dict[str, Any]:
    """Delegate only post-walk lineage materialization to the 582 resolver."""

    symbol = "iter_candidate_lineage_from_selected_trace"
    original = getattr(science_graph_trace, symbol, None)
    semantic_iterator = getattr(semantic_graph_trace, symbol, None)
    original_collapse = getattr(
        semantic_graph_trace, "collapse_semantic_transition_aliases", None
    )
    if not callable(original) or not callable(semantic_iterator):
        raise ValueError("Frozen science lineage iterator is not patchable")
    if not callable(original_collapse):
        raise ValueError("Reviewed semantic alias resolver is absent")
    state: dict[str, Any] = {
        "invocations": [],
        "active_sequence": None,
    }

    def recording_collapse(*args: Any, **kwargs: Any) -> Any:
        resolution = original_collapse(*args, **kwargs)
        active = state.get("active_sequence")
        if not isinstance(active, list):
            raise RuntimeError(
                "Semantic resolver executed outside delegated lineage finalization"
            )
        active.append(resolution.audit_dict())
        return resolution

    def delegated_iterator(
        payload: Mapping[str, Any],
        selected_events: Any,
        *,
        source_graphs_by_parent_id: Mapping[str, Any] | None = None,
        include_actions: bool = True,
        recovery_audit: dict[str, Any] | None = None,
    ) -> Any:
        active_recovery_audit: dict[str, Any] = (
            recovery_audit if recovery_audit is not None else {}
        )
        semantic_sequence: list[dict[str, Any]] = []
        state["active_sequence"] = semantic_sequence
        row_count = 0
        unresolved_row_count = 0
        try:
            for row in semantic_iterator(
                payload,
                selected_events,
                source_graphs_by_parent_id=source_graphs_by_parent_id,
                include_actions=include_actions,
                recovery_audit=active_recovery_audit,
            ):
                row_count += 1
                unresolved_row_count += int(
                    row.get("action_lineage_resolved") is not True
                )
                yield row
        except BaseException as exc:
            invocation = {
                "status": "FAIL",
                "row_count": row_count,
                "unresolved_row_count": unresolved_row_count,
                "include_actions": bool(include_actions),
                "semantic_transition_count": len(semantic_sequence),
                "semantic_transition_sequence_sha256": _stable_json_sha256(
                    semantic_sequence
                ),
                "recovery_audit": active_recovery_audit,
                "error_class": type(exc).__name__,
                "error": str(exc),
            }
            state["invocations"].append(invocation)
            _write_semantic_finalizer_receipt(
                receipt_path,
                status="FAIL",
                binding=binding,
                role=role,
                science_project_commit=science_project_commit,
                invocations=state["invocations"],
            )
            raise
        else:
            invocation = {
                "status": "PASS",
                "row_count": row_count,
                "unresolved_row_count": unresolved_row_count,
                "include_actions": bool(include_actions),
                "semantic_transition_count": len(semantic_sequence),
                "semantic_transition_sequence_sha256": _stable_json_sha256(
                    semantic_sequence
                ),
                "semantic_transition_alias_classes": [
                    {"sequence_index": index, **row}
                    for index, row in enumerate(semantic_sequence)
                    if int(row.get("raw_alias_count", 0)) > 0
                    or int(row.get("semantic_class_count", 0)) > 1
                ],
                "recovery_audit": active_recovery_audit,
            }
            state["invocations"].append(invocation)
            _write_semantic_finalizer_receipt(
                receipt_path,
                status="PASS",
                binding=binding,
                role=role,
                science_project_commit=science_project_commit,
                invocations=state["invocations"],
            )
        finally:
            state["active_sequence"] = None

    semantic_graph_trace.collapse_semantic_transition_aliases = recording_collapse
    setattr(science_graph_trace, symbol, delegated_iterator)
    return state


def _write_semantic_finalizer_receipt(
    path: Path,
    *,
    status: str,
    binding: Mapping[str, Any],
    role: str,
    science_project_commit: str,
    invocations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    invocation_rows = [dict(row) for row in invocations]
    value: dict[str, Any] = {
        "schema_version": SEMANTIC_FINALIZER_SCHEMA,
        "status": status,
        "role": role,
        "science_project_commit": science_project_commit,
        "source_algorithm_commit": SOURCE_COMMIT,
        "semantic_finalizer_binding": dict(binding),
        "patched_science_symbols": [
            "src.baselines.comrecgc.graph_trace."
            "iter_candidate_lineage_from_selected_trace"
        ],
        "patched_generation_symbols": [],
        "random_walk_patched": False,
        "candidate_generation_patched": False,
        "candidate_universe_patched": False,
        "delegation_scope": "post_walk_lineage_materialization_only",
        "invocation_count": len(invocation_rows),
        "invocations": invocation_rows,
        "semantic_transition_count": sum(
            int(row.get("semantic_transition_count", 0))
            for row in invocation_rows
        ),
        "semantic_transition_alias_event_count": sum(
            len(row.get("semantic_transition_alias_classes") or [])
            for row in invocation_rows
        ),
        "all_lineage_rows_resolved": all(
            int(row.get("unresolved_row_count", -1)) == 0
            and int((row.get("recovery_audit") or {}).get(
                "semantic_transition_failure_count", 0
            )) == 0
            and int((row.get("recovery_audit") or {}).get(
                "semantic_transition_excluded_count", 0
            )) == 0
            for row in invocation_rows
        ),
    }
    value["receipt_sha256"] = _stable_json_sha256(value)
    _atomic_json(path, value)
    return value


def _validate_semantic_finalizer_receipt(
    path: Path,
    *,
    role: str,
    science_project_commit: str,
) -> dict[str, Any]:
    value = _json(path)
    failures: list[str] = []
    required = {
        "schema_version": SEMANTIC_FINALIZER_SCHEMA,
        "status": "PASS",
        "role": role,
        "science_project_commit": science_project_commit,
        "source_algorithm_commit": SOURCE_COMMIT,
        "patched_generation_symbols": [],
        "random_walk_patched": False,
        "candidate_generation_patched": False,
        "candidate_universe_patched": False,
        "delegation_scope": "post_walk_lineage_materialization_only",
        "all_lineage_rows_resolved": True,
    }
    failures.extend(
        key for key, expected in required.items() if value.get(key) != expected
    )
    binding = value.get("semantic_finalizer_binding")
    if not isinstance(binding, Mapping) or any(
        binding.get(key) != expected
        for key, expected in {
            "status": "PASS",
            "commit": SEMANTIC_FINALIZER_COMMIT,
            "graph_trace_sha256": SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256,
            "resolver_symbol": "collapse_semantic_transition_aliases",
            "lineage_symbol": "iter_candidate_lineage_from_selected_trace",
            "module_namespaces_disjoint": True,
        }.items()
    ):
        failures.append("semantic_finalizer_binding")
    invocations = value.get("invocations")
    if (
        not isinstance(invocations, list)
        or not invocations
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "PASS"
            or int(row.get("semantic_transition_count", 0)) <= 0
            or int(row.get("unresolved_row_count", -1)) != 0
            for row in invocations
        )
    ):
        failures.append("invocations")
    claimed = value.get("receipt_sha256")
    if claimed != _stable_json_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        failures.append("receipt_sha256")
    if failures:
        raise ValueError(
            "Semantic lineage finalizer receipt failed closed: "
            + ",".join(sorted(set(failures)))
        )
    return value


def _run_one(args: argparse.Namespace) -> int:
    science_root = _absolute(args.science_project_root)
    output = _absolute(args.output_root, exists=False)
    expected_commit = str(args.expected_commit)
    if _git_head(science_root) != expected_commit:
        raise ValueError("Scientific worktree commit changed")
    _install_science_root(science_root)
    import numpy as np
    import torch

    from src.baselines.comrecgc.contracts import (
        GenerationParameters,
        sha256_file,
        stable_json_sha256,
    )
    from src.baselines.comrecgc.runtime import run_project_generation
    science_graph_trace = importlib.import_module(
        "src.baselines.comrecgc.graph_trace"
    )

    semantic_graph_trace, semantic_binding = _load_semantic_finalizer(
        _absolute(args.semantic_finalizer_project_root)
    )
    semantic_receipt_path = output / "semantic_lineage_finalizer_receipt.json"
    _install_semantic_lineage_finalizer(
        science_graph_trace=science_graph_trace,
        semantic_graph_trace=semantic_graph_trace,
        binding=semantic_binding,
        receipt_path=semantic_receipt_path,
        role=str(args.role),
        science_project_commit=expected_commit,
    )

    expected_full = GenerationParameters.for_mode("full")
    parameters = replace(expected_full, steps=STEPS)

    def _validate_prefix(self: Any, mode: str) -> None:
        if mode != "full" or self != parameters:
            raise ValueError("Diagnostic generation parameters changed")

    GenerationParameters.validate = _validate_prefix
    signature = inspect.signature(run_project_generation)
    checkpoint_capable = "checkpoint_root" in signature.parameters
    if args.role == "legacy" and checkpoint_capable:
        raise ValueError("Legacy worktree unexpectedly contains checkpoint instrumentation")
    if args.role == "instrumented" and not checkpoint_capable:
        raise ValueError("Instrumented worktree has no checkpoint implementation")
    kwargs: dict[str, Any] = {
        "project_root": science_root,
        "upstream_root": _absolute(args.upstream_root),
        "dataset": "mutagenicity",
        "dataset_dir": _absolute(args.dataset_dir),
        "source_csv": None,
        "gnn_checkpoint": _absolute(args.gnn_checkpoint),
        "distance_checkpoint": _absolute(args.distance_checkpoint),
        "output_dir": output,
        "mode": "full",
        "parent_limit": int(args.parent_limit),
        "parameters": parameters,
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "resume": bool(args.resume),
        "trace_output_dir": output / "trace",
        "parity_reference_path": None,
        "graph_state_dir": output / "graph_state",
        "storage_guard_root": output,
        "storage_check_every_steps": 250,
        "storage_min_free_bytes": 50 * 1024**3,
        "storage_min_free_ratio": 0.02,
        "storage_min_free_inodes": 100_000,
    }
    if checkpoint_capable:
        from src.baselines.comrecgc.generation_checkpoint import (
            scientific_command_sha256,
        )

        scientific_argv = (
            "mut_checkpoint_instrumentation_equivalence_v1",
            "dataset=mutagenicity",
            f"source_algorithm_commit={SOURCE_COMMIT}",
            f"execution_commit={expected_commit}",
            f"parent_limit={int(args.parent_limit)}",
            f"steps={STEPS}",
            f"seed={parameters.seed}",
            f"upstream={Path(args.upstream_root).resolve()}",
            f"dataset_dir={Path(args.dataset_dir).resolve()}",
            f"gnn={Path(args.gnn_checkpoint).resolve()}",
            f"distance={Path(args.distance_checkpoint).resolve()}",
            f"output={output}",
        )
        kwargs.update(
            {
                "checkpoint_root": output / "generation_checkpoints",
                "checkpoint_mirror_root": _absolute(
                    args.checkpoint_mirror_root, exists=False
                ),
                "checkpoint_interval_steps": 250,
                "checkpoint_keep_last": 2,
                "progress_interval_steps": 25,
                "scientific_argv": scientific_argv,
                "command_sha256": scientific_command_sha256(scientific_argv),
            }
        )
    manifest = run_project_generation(**kwargs)
    semantic_receipt = _validate_semantic_finalizer_receipt(
        semantic_receipt_path,
        role=str(args.role),
        science_project_commit=expected_commit,
    )
    rng = _rng_payload(torch, np)
    diagnostic = {
        "schema_version": "mut_checkpoint_instrumentation_prefix_v1",
        "status": "PASS",
        "role": args.role,
        "source_algorithm_commit": SOURCE_COMMIT,
        "science_project_commit": expected_commit,
        "checkpoint_capable": checkpoint_capable,
        "resumed": bool(args.resume),
        "parameters": asdict(parameters),
        "parent_limit": int(args.parent_limit),
        "dataset": "mutagenicity",
        "trace_enabled_for_equivalence_only": True,
        "paper_eligible": False,
        "calibration_loaded": manifest.get("calibration_loaded"),
        "test_loaded": manifest.get("test_loaded"),
        "counterfactuals_sha256": sha256_file(output / "counterfactuals.pt"),
        "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        "semantic_lineage_finalizer_receipt": str(semantic_receipt_path),
        "semantic_lineage_finalizer_receipt_sha256": hashlib.sha256(
            semantic_receipt_path.read_bytes()
        ).hexdigest(),
        "semantic_lineage_finalizer_commit": SEMANTIC_FINALIZER_COMMIT,
        "semantic_lineage_finalizer_graph_trace_sha256": (
            SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256
        ),
        "semantic_transition_count": semantic_receipt[
            "semantic_transition_count"
        ],
        "semantic_transition_alias_event_count": semantic_receipt[
            "semantic_transition_alias_event_count"
        ],
        "rng_state_sha256": stable_json_sha256(rng),
        "rng_state": rng,
    }
    if (
        diagnostic["calibration_loaded"] is not False
        or diagnostic["test_loaded"] is not False
    ):
        raise ValueError("Diagnostic prefix loaded calibration/test data")
    _atomic_json(output / "DIAGNOSTIC_ONLY.json", diagnostic)
    print(json.dumps(diagnostic, sort_keys=True))
    return 0


def _source_inventory(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.exists():
            files[relative] = {"present": False}
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Scientific source is not one physical file: {path}")
        source = path.read_bytes()
        definitions: dict[str, str] = {}
        if path.suffix == ".py":
            tree = ast.parse(source.decode("utf-8"), filename=str(path))
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
    payload["inventory_sha256"] = _stable_json_sha256(
        {key: value for key, value in payload.items() if key != "project_root"}
    )
    return payload


def _source_delta_audit(
    legacy: Mapping[str, Any], instrumented: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify every reviewed scientific-source delta, without hiding it."""

    unchanged = {
        "src/baselines/comrecgc/contracts.py",
        "src/baselines/comrecgc/project_dataset.py",
        "src/baselines/comrecgc/model_adapter.py",
        "src/baselines/comrecgc/upstream.py",
    }
    checkpoint_new = {
        "src/baselines/comrecgc/generation_checkpoint.py",
        "src/baselines/comrecgc/generation_loop.py",
    }
    allowed_changed_definitions = {
        "scripts/baselines/comrecgc/run_generation.py": {
            "build_parser",
            "main",
            "_redact_cli_value",
            "canonical_scientific_argv",
        },
        "src/baselines/comrecgc/runtime.py": {
            "PatchedRuntimeHandles",
            "patched_official_runtime",
            "run_project_generation",
            "_checkpoint_algorithm_state",
            "_load_persistent_resolved_config",
            "_persistent_resolved_config_paths",
            "_progress_payload",
            "_publish_persistent_resolved_config",
            "_resolved_config_content_sha256",
            "_restore_runtime_checkpoint_state",
            "_runtime_checkpoint_provenance",
            "_runtime_environment",
            "_validate_resolved_config_binding",
        },
        "src/baselines/comrecgc/graph_trace.py": {
            "ActionTraceRecorder",
            "_lineage_recovery_context",
            "_official_single_edit_diagnostic",
            "enumerate_official_single_edits",
            "infer_official_single_edit",
            "iter_candidate_lineage_from_selected_trace",
            "recover_candidate_lineage_from_selected_trace",
        },
        "src/baselines/comrecgc/live_graph_state.py": {
            "AuthoritativeGraphStore",
            "LiveGraphMap",
            "LiveGraphState",
        },
        "src/baselines/comrecgc/transition_cache.py": {
            "CompactMoveScopedTransitionMap",
        },
        "src/baselines/comrecgc/storage_guard.py": {"StorageGuard"},
    }
    legacy_files = legacy.get("files") or {}
    instrumented_files = instrumented.get("files") or {}
    failures: list[str] = []
    details: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        left = legacy_files.get(relative) or {}
        right = instrumented_files.get(relative) or {}
        if relative in unchanged:
            exact = left.get("present") is True and left == right
            details[relative] = {"classification": "scientific_source_unchanged", "exact": exact}
            if not exact:
                failures.append(f"unexpected_scientific_source_change:{relative}")
            continue
        if relative in checkpoint_new:
            exact = left.get("present") is False and right.get("present") is True
            details[relative] = {"classification": "checkpoint_resume_module_added", "exact": exact}
            if not exact:
                failures.append(f"checkpoint_module_shape:{relative}")
            continue
        left_defs = left.get("top_level_definition_ast_sha256") or {}
        right_defs = right.get("top_level_definition_ast_sha256") or {}
        changed = {
            name
            for name in set(left_defs) & set(right_defs)
            if left_defs[name] != right_defs[name]
        }
        added = set(right_defs) - set(left_defs)
        removed = set(left_defs) - set(right_defs)
        delta = changed | added | removed
        allowed = allowed_changed_definitions.get(relative, set())
        exact = (
            left.get("present") is True
            and right.get("present") is True
            and not removed
            and bool(delta)
            and delta <= allowed
        )
        details[relative] = {
            "classification": (
                "checkpoint_runtime_or_operational_trace_support_reviewed"
            ),
            "changed_definitions": sorted(changed),
            "added_definitions": sorted(added),
            "removed_definitions": sorted(removed),
            "allowed_definitions": sorted(allowed),
            "exact": exact,
        }
        if not exact:
            failures.append(f"unreviewed_source_delta:{relative}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "all_source_deltas_visible": True,
        "source_text_identity_required": False,
        "claimed_effect_for_mut_traceoff": "checkpoint_and_exact_resume_only",
        "behavioral_equivalence_required": True,
        "details": details,
        "failures": failures,
    }


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _run_child(command: Sequence[str], *, log: Path, environment: Mapping[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            list(command),
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Diagnostic child failed ({result.returncode}): {log}")


def _interrupt_at_checkpoint(
    command: Sequence[str],
    *,
    log: Path,
    environment: Mapping[str, str],
    mirror_root: Path,
    proof_path: Path,
) -> None:
    """Exercise a real SIGTERM-at-completed-step resume boundary once."""

    checkpoint_marker = (
        mirror_root / "step-000000000250/_CHECKPOINT_MIRRORED.json"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 72 * 60 * 60
        while process.poll() is None and not checkpoint_marker.is_file():
            if time.monotonic() >= deadline:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=120)
                raise TimeoutError("Instrumented prefix did not reach step 250 in 72 hours")
            time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(
                "Instrumented prefix exited before the controlled resume boundary: "
                f"returncode={process.returncode}"
            )
        marker_sha256 = hashlib.sha256(checkpoint_marker.read_bytes()).hexdigest()
        process.send_signal(signal.SIGTERM)
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Instrumented diagnostic ignored SIGTERM; no SIGKILL is permitted"
            ) from exc
    if returncode not in {-signal.SIGTERM, 128 + signal.SIGTERM}:
        raise RuntimeError(
            f"Controlled instrumentation stop returned unexpectedly: {returncode}"
        )
    output_root = mirror_root.parent / "instrumented"
    if (output_root / "_RUN_COMPLETE.json").exists():
        raise RuntimeError("Instrumented diagnostic completed before resume was exercised")
    _atomic_json(
        proof_path,
        {
            "schema_version": "mut_checkpoint_instrumentation_interrupt_v1",
            "status": "PASS",
            "signal": "SIGTERM",
            "returncode": returncode,
            "completed_checkpoint_step": 250,
            "checkpoint_marker": str(checkpoint_marker),
            "checkpoint_marker_sha256": marker_sha256,
            "run_complete_absent_after_interrupt": True,
            "created_at_unix": time.time(),
        },
    )


def _run_pair(args: argparse.Namespace) -> int:
    controller_root = Path(__file__).resolve().parents[2]
    legacy_project = _absolute(args.legacy_project_root)
    execution_project = _absolute(args.execution_project_root)
    semantic_finalizer_project = _absolute(args.semantic_finalizer_project_root)
    if _git_head(legacy_project) != SOURCE_COMMIT:
        raise ValueError("Legacy project is not the exact traced-source commit")
    execution_commit = str(args.execution_commit)
    if execution_commit != INSTRUMENTATION_COMMIT:
        raise ValueError("Execution commit is not the reviewed checkpoint release")
    if _git_head(execution_project) != execution_commit:
        raise ValueError("Instrumentation execution commit changed")
    if _git_head(semantic_finalizer_project) != SEMANTIC_FINALIZER_COMMIT:
        raise ValueError("Semantic finalizer worktree commit changed")
    legacy_source_inventory = _source_inventory(legacy_project)
    instrumentation_source_inventory = _source_inventory(execution_project)
    if (
        args.expected_legacy_inventory_sha256 != LEGACY_SOURCE_INVENTORY_SHA256
        or args.expected_instrumentation_inventory_sha256
        != INSTRUMENTATION_SOURCE_INVENTORY_SHA256
    ):
        raise ValueError("Controller supplied an unreviewed scientific inventory")
    if (
        legacy_source_inventory.get("inventory_sha256")
        != args.expected_legacy_inventory_sha256
    ):
        raise ValueError("Legacy scientific source inventory changed")
    if (
        instrumentation_source_inventory.get("inventory_sha256")
        != args.expected_instrumentation_inventory_sha256
    ):
        raise ValueError("Instrumentation scientific source inventory changed")
    source_delta = _source_delta_audit(
        legacy_source_inventory, instrumentation_source_inventory
    )
    if source_delta["status"] != "PASS":
        raise ValueError(
            "Scientific source delta is outside the reviewed instrumentation boundary: "
            f"{source_delta['failures']}"
        )
    run_root = _absolute(args.run_root, exists=False)
    gate_root = _absolute(args.output_dir, exists=False)
    if gate_root.exists() or gate_root.is_symlink():
        raise FileExistsError(f"Equivalence gate output must be fresh: {gate_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    adopted_legacy = (
        _absolute(args.adopt_complete_legacy_root)
        if args.adopt_complete_legacy_root
        else None
    )
    legacy = adopted_legacy or (run_root / "legacy")
    instrumented = run_root / "instrumented"
    mirror = run_root / "instrumented-checkpoint-mirror"
    interruption_proof_path = run_root / "instrumented-interruption-proof.json"
    base = [
        str(_absolute(args.python)),
        str(controller_root / "scripts/autodl/run_mut_checkpoint_instrumentation_equivalence.py"),
        "run-one",
        "--upstream-root",
        str(_absolute(args.upstream_root)),
        "--dataset-dir",
        str(_absolute(args.dataset_dir)),
        "--gnn-checkpoint",
        str(_absolute(args.gnn_checkpoint)),
        "--distance-checkpoint",
        str(_absolute(args.distance_checkpoint)),
        "--parent-limit",
        str(int(args.parent_limit)),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
        "--semantic-finalizer-project-root",
        str(semantic_finalizer_project),
    ]
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if adopted_legacy is not None:
        if not (legacy / "_RUN_COMPLETE.json").is_file():
            raise ValueError("Adopted legacy arm is not complete")
        required_legacy_files = (
            "run_manifest.json",
            "DIAGNOSTIC_ONLY.json",
            "counterfactuals.pt",
            "semantic_lineage_finalizer_receipt.json",
            "trace/selected_action_trace_manifest.json",
        )
        missing = [
            relative
            for relative in required_legacy_files
            if not (legacy / relative).is_file()
            or (legacy / relative).is_symlink()
        ]
        if missing:
            raise ValueError(
                "Adopted legacy arm is incomplete: " + ",".join(missing)
            )
        adoption = {
            "schema_version": "mut_complete_legacy_arm_adoption_v1",
            "status": "PASS",
            "role": "legacy",
            "source_root": str(legacy),
            "source_project_commit": SOURCE_COMMIT,
            "source_run_complete_sha256": _sha256_path(
                legacy / "_RUN_COMPLETE.json"
            ),
            "source_run_manifest_sha256": _sha256_path(
                legacy / "run_manifest.json"
            ),
            "source_diagnostic_sha256": _sha256_path(
                legacy / "DIAGNOSTIC_ONLY.json"
            ),
            "source_counterfactuals_sha256": _sha256_path(
                legacy / "counterfactuals.pt"
            ),
            "source_trace_manifest_sha256": _sha256_path(
                legacy / "trace/selected_action_trace_manifest.json"
            ),
            "source_arm_recomputed": False,
            "instrumented_arm_restart_step": 0,
            "source_root_write_allowed": False,
        }
        adoption["receipt_sha256"] = _stable_json_sha256(adoption)
        _atomic_json(run_root / "legacy_arm_adoption.json", adoption)
    elif not (legacy / "_RUN_COMPLETE.json").is_file():
        if legacy.exists():
            raise FileExistsError("Incomplete legacy prefix is not resumable")
        _run_child(
            [
                *base,
                "--role",
                "legacy",
                "--science-project-root",
                str(legacy_project),
                "--expected-commit",
                SOURCE_COMMIT,
                "--output-root",
                str(legacy),
            ],
            log=run_root / "legacy.log",
            environment=environment,
        )
    instrumented_command = [
            *base,
            "--role",
            "instrumented",
            "--science-project-root",
            str(execution_project),
            "--expected-commit",
            execution_commit,
            "--output-root",
            str(instrumented),
            "--checkpoint-mirror-root",
            str(mirror),
        ]
    if not instrumented.exists():
        if interruption_proof_path.exists():
            raise ValueError("Interruption proof exists without its scientific root")
        _interrupt_at_checkpoint(
            instrumented_command,
            log=run_root / "instrumented.log",
            environment=environment,
            mirror_root=mirror,
            proof_path=interruption_proof_path,
        )
    if not (instrumented / "_RUN_COMPLETE.json").is_file():
        if not interruption_proof_path.is_file():
            raise ValueError(
                "Incomplete instrumented prefix lacks controlled-interruption proof"
            )
        _run_child(
            [*instrumented_command, "--resume"],
            log=run_root / "instrumented.log",
            environment=environment,
        )
    # Audit with the immutable controller release.  The 664 scientific
    # worktree intentionally predates the generic BACE equivalence module.
    sys.path.insert(0, str(controller_root))
    from src.baselines.comrecgc.equivalence import _payload_equivalence
    from src.baselines.comrecgc.generation_checkpoint import (
        MIRRORED_FILENAME,
        validate_generation_checkpoint,
    )
    from src.baselines.comrecgc.graph_trace import load_selected_trace
    from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256

    legacy_manifest = _json(legacy / "run_manifest.json")
    instrumented_manifest = _json(instrumented / "run_manifest.json")
    legacy_diagnostic = _json(legacy / "DIAGNOSTIC_ONLY.json")
    instrumented_diagnostic = _json(instrumented / "DIAGNOSTIC_ONLY.json")
    legacy_semantic_receipt_path = (
        legacy / "semantic_lineage_finalizer_receipt.json"
    )
    instrumented_semantic_receipt_path = (
        instrumented / "semantic_lineage_finalizer_receipt.json"
    )
    legacy_semantic_receipt = _validate_semantic_finalizer_receipt(
        legacy_semantic_receipt_path,
        role="legacy",
        science_project_commit=SOURCE_COMMIT,
    )
    instrumented_semantic_receipt = _validate_semantic_finalizer_receipt(
        instrumented_semantic_receipt_path,
        role="instrumented",
        science_project_commit=execution_commit,
    )
    interruption_proof = _json(interruption_proof_path)
    interruption_marker = Path(str(interruption_proof.get("checkpoint_marker") or ""))
    interruption_marker_valid = (
        interruption_marker
        == mirror / "step-000000000250/_CHECKPOINT_MIRRORED.json"
        and interruption_marker.is_file()
        and not interruption_marker.is_symlink()
        and hashlib.sha256(interruption_marker.read_bytes()).hexdigest()
        == interruption_proof.get("checkpoint_marker_sha256")
    )
    identity_fields = (
        "dataset",
        "upstream_commit",
        "generation_parent_ids_sha256",
        "cf_mode",
        "parent_limit",
    )
    failures = [
        key
        for key in identity_fields
        if legacy_manifest.get(key) != instrumented_manifest.get(key)
    ]
    for key in (
        "parameters",
        "dataset_audit",
        "internal_prediction_counts",
        "gnn",
        "distance_model",
    ):
        if legacy_manifest.get(key) != instrumented_manifest.get(key):
            failures.append(key)
    if legacy_manifest.get("project_commit") != SOURCE_COMMIT:
        failures.append("legacy_project_commit")
    if instrumented_manifest.get("project_commit") != execution_commit:
        failures.append("instrumented_project_commit")
    if legacy_manifest.get("trace_enabled") is not True or instrumented_manifest.get(
        "trace_enabled"
    ) is not True:
        failures.append("diagnostic_trace")
    if legacy_diagnostic.get("rng_state_sha256") != instrumented_diagnostic.get(
        "rng_state_sha256"
    ):
        failures.append("rng_state")
    legacy_semantic_sequences = [
        row["semantic_transition_sequence_sha256"]
        for row in legacy_semantic_receipt["invocations"]
    ]
    instrumented_semantic_sequences = [
        row["semantic_transition_sequence_sha256"]
        for row in instrumented_semantic_receipt["invocations"]
    ]
    semantic_transition_sequence_exact = (
        legacy_semantic_sequences == instrumented_semantic_sequences
    )
    if not semantic_transition_sequence_exact:
        failures.append("semantic_transition_sequence")
    if (
        legacy_diagnostic.get("resumed") is not False
        or instrumented_diagnostic.get("resumed") is not True
        or interruption_proof.get("status") != "PASS"
        or interruption_proof.get("signal") != "SIGTERM"
        or int(interruption_proof.get("completed_checkpoint_step", -1)) != 250
        or interruption_proof.get("run_complete_absent_after_interrupt") is not True
        or not interruption_marker_valid
    ):
        failures.append("checkpoint_resume_exercise")
    payload_equivalence = _payload_equivalence(
        _torch_load(legacy / "counterfactuals.pt"),
        _torch_load(instrumented / "counterfactuals.pt"),
    )
    failures.extend(payload_equivalence["failures"])
    legacy_trace = load_selected_trace(
        legacy / "trace/selected_action_trace_manifest.json"
    )
    instrumented_trace = load_selected_trace(
        instrumented / "trace/selected_action_trace_manifest.json"
    )
    trace_exact = legacy_trace == instrumented_trace
    if not trace_exact:
        failures.append("step_action_trace")
    local_checkpoint = validate_generation_checkpoint(
        instrumented / "generation_checkpoints", expected_completed_step=STEPS
    )
    mirror_checkpoint = validate_generation_checkpoint(
        mirror,
        expected_provenance=local_checkpoint.provenance_fingerprints,
        expected_scientific_argv=local_checkpoint.scientific_argv,
        expected_command_sha256=local_checkpoint.command_sha256,
        expected_total_steps=STEPS,
        expected_completed_step=STEPS,
    )
    marker = _json(mirror_checkpoint.checkpoint_dir / MIRRORED_FILENAME)
    if (
        mirror_checkpoint.checkpoint_digest != local_checkpoint.checkpoint_digest
        or marker.get("checkpoint_mirrored") is not True
        or marker.get("checkpoint_digest") != local_checkpoint.checkpoint_digest
    ):
        failures.append("checkpoint_mirror")
    source_audit = {
        "legacy": legacy_source_inventory,
        "instrumented": instrumentation_source_inventory,
        "delta_audit": source_delta,
        "intended_scientific_effect": "checkpoint_resume_instrumentation_only",
        "source_text_identity_required": False,
        "runtime_equivalence_required": True,
    }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": STEPS,
        "seed": 0,
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_instrumentation_commit": execution_commit,
        "equivalence_auditor_commit": _git_head(controller_root),
        "legacy_root": str(legacy),
        "instrumented_root": str(instrumented),
        "legacy_payload_sha256": sha256_file(legacy / "counterfactuals.pt"),
        "instrumented_payload_sha256": sha256_file(
            instrumented / "counterfactuals.pt"
        ),
        "payload_equivalence": payload_equivalence,
        "step_action_trace_exact": trace_exact,
        "step_action_count": len(legacy_trace),
        "rng_state_exact": legacy_diagnostic.get("rng_state_sha256")
        == instrumented_diagnostic.get("rng_state_sha256"),
        "checkpoint_digest": local_checkpoint.checkpoint_digest,
        "checkpoint_mirror_verified": "checkpoint_mirror" not in failures,
        "checkpoint_resume_exercised": (
            "checkpoint_resume_exercise" not in failures
        ),
        "checkpoint_interruption_proof": str(interruption_proof_path),
        "checkpoint_interruption_proof_sha256": sha256_file(
            interruption_proof_path
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "source_audit": source_audit,
        "semantic_lineage_finalizer": {
            "status": "PASS",
            "commit": SEMANTIC_FINALIZER_COMMIT,
            "project_root": str(semantic_finalizer_project),
            "graph_trace_sha256": SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256,
            "delegation_scope": "post_walk_lineage_materialization_only",
            "generation_algorithm_changed": False,
            "candidate_universe_changed": False,
            "legacy_receipt": str(legacy_semantic_receipt_path),
            "legacy_receipt_sha256": hashlib.sha256(
                legacy_semantic_receipt_path.read_bytes()
            ).hexdigest(),
            "instrumented_receipt": str(instrumented_semantic_receipt_path),
            "instrumented_receipt_sha256": hashlib.sha256(
                instrumented_semantic_receipt_path.read_bytes()
            ).hexdigest(),
            "legacy_semantic_transition_alias_event_count": (
                legacy_semantic_receipt[
                    "semantic_transition_alias_event_count"
                ]
            ),
            "instrumented_semantic_transition_alias_event_count": (
                instrumented_semantic_receipt[
                    "semantic_transition_alias_event_count"
                ]
            ),
            "semantic_transition_sequence_exact": (
                semantic_transition_sequence_exact
            ),
            "legacy_semantic_transition_sequence_sha256": (
                legacy_semantic_sequences
            ),
            "instrumented_semantic_transition_sequence_sha256": (
                instrumented_semantic_sequences
            ),
        },
        "failures": failures,
    }
    result["summary_sha256"] = stable_json_sha256(result)
    gate_root.mkdir(parents=True)
    _atomic_json(gate_root / "equivalence.json", result)
    if failures:
        _atomic_json(gate_root / "FAIL.json", result)
        raise RuntimeError(f"Mut checkpoint instrumentation differs: {failures}")
    descriptor = os.open(
        gate_root / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(result, sort_keys=True))
    print("[MUT_CHECKPOINT_INSTRUMENTATION_EQUIVALENCE_PASS]", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    one = commands.add_parser("run-one")
    one.add_argument("--role", choices=("legacy", "instrumented"), required=True)
    one.add_argument("--science-project-root", required=True)
    one.add_argument("--expected-commit", required=True)
    one.add_argument("--output-root", required=True)
    one.add_argument("--checkpoint-mirror-root")
    one.add_argument("--resume", action="store_true")
    pair = commands.add_parser("run-pair")
    pair.add_argument("--python", required=True)
    pair.add_argument("--legacy-project-root", required=True)
    pair.add_argument("--execution-project-root", required=True)
    pair.add_argument("--execution-commit", required=True)
    pair.add_argument("--expected-legacy-inventory-sha256", required=True)
    pair.add_argument("--expected-instrumentation-inventory-sha256", required=True)
    pair.add_argument("--run-root", required=True)
    pair.add_argument("--output-dir", required=True)
    pair.add_argument(
        "--adopt-complete-legacy-root",
        help=(
            "read-only completed legacy/A arm to adopt while restarting only the "
            "instrumented/B arm from step zero"
        ),
    )
    for value in (one, pair):
        value.add_argument("--upstream-root", required=True)
        value.add_argument("--dataset-dir", required=True)
        value.add_argument("--gnn-checkpoint", required=True)
        value.add_argument("--distance-checkpoint", required=True)
        value.add_argument("--parent-limit", type=int, default=1448)
        value.add_argument("--device", default="cuda:0")
        value.add_argument("--batch-size", type=int, default=128)
        value.add_argument("--semantic-finalizer-project-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_one(args) if args.action == "run-one" else _run_pair(args)


if __name__ == "__main__":
    raise SystemExit(main())
