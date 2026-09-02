from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import tempfile

import pytest

import scripts.autodl.manage_mut_traceoff_parity_v1 as manage
import scripts.autodl.run_mut_checkpoint_instrumentation_equivalence as equiv
from scripts.autodl.run_four_gpu_recovery_controller import ControllerError
from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256
import src.utils.autodl_mut_traceoff_parity_v1 as mut
from src.utils.autodl_four_by_four_repair import RepairManifestError


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _text(path: Path, value: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_semantic_finalizer_git_status_ignores_only_untracked_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "reviewed-finalizer"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    tracked = _text(repository / "graph_trace.py", "reviewed\n")
    subprocess.run(["git", "-C", str(repository), "add", "graph_trace.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "reviewed"],
        check=True,
    )

    _text(repository / "diagnostic-receipt.json", "{}\n")
    assert equiv._git_science_status(repository) == ""

    tracked.write_text("changed\n", encoding="utf-8")
    assert "graph_trace.py" in equiv._git_science_status(repository)


def test_semantic_finalizer_loader_uses_private_582_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(equiv, "_git_head", lambda _root: equiv.SEMANTIC_FINALIZER_COMMIT)
    monkeypatch.setattr(equiv, "_git_science_status", lambda _root: "")
    package_name = "_mut_semantic_finalizer_582bc4b"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name)
    try:
        module, binding = equiv._load_semantic_finalizer(repository)
        assert module.__name__ == f"{package_name}.graph_trace"
        assert binding["commit"] == equiv.SEMANTIC_FINALIZER_COMMIT
        assert binding["module_namespaces_disjoint"] is True
        assert module.SemanticTransitionKey.__module__.startswith(package_name)
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name)


def test_checkpoint_prefix_delegates_only_lineage_to_582_semantic_resolver(
    tmp_path: Path,
) -> None:
    science = ModuleType("frozen_science_graph_trace")
    semantic = ModuleType("reviewed_semantic_graph_trace")

    def forbidden_old_iterator(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("legacy raw-cardinality finalizer must be replaced")

    class Resolution:
        def audit_dict(self) -> dict[str, object]:
            return {
                "status": "SELECTED_AUTHORITATIVE_HISTORIC_PIN",
                "semantic_key": {
                    "pinned_upstream_graph_hash": "a" * 64,
                    "normalized_action_type": "NR",
                    "normalized_single_edit_signature": "b" * 64,
                    "downstream_graph_hash": "c" * 64,
                    "action_parameter_digest": "d" * 64,
                    "source_head_id": "2",
                    "generation_step": 139,
                },
                "action": ["NR", 3, 3],
                "raw_candidate_count": 3,
                "raw_unique_candidate_count": 3,
                "raw_alias_count": 2,
                "semantic_class_count": 1,
                "lineage_valid_candidate_count": 1,
                "authoritative_historic_pin_used": True,
                "representative_replaced": False,
            }

    def collapse(*_args: object, **_kwargs: object) -> Resolution:
        return Resolution()

    def current_iterator(
        _payload: object,
        _selected_events: object,
        *,
        source_graphs_by_parent_id: object = None,
        include_actions: bool = True,
        recovery_audit: dict[str, object],
    ) -> object:
        del source_graphs_by_parent_id, include_actions
        semantic.collapse_semantic_transition_aliases()
        recovery_audit.update(
            {
                "semantic_transition_failure_count": 0,
                "semantic_transition_excluded_count": 0,
            }
        )
        yield {"candidate_index": 0, "action_lineage_resolved": True}

    science.iter_candidate_lineage_from_selected_trace = forbidden_old_iterator
    semantic.collapse_semantic_transition_aliases = collapse
    semantic.iter_candidate_lineage_from_selected_trace = current_iterator
    receipt = tmp_path / "semantic_lineage_finalizer_receipt.json"
    binding = {
        "status": "PASS",
        "commit": equiv.SEMANTIC_FINALIZER_COMMIT,
        "graph_trace_sha256": equiv.SEMANTIC_FINALIZER_GRAPH_TRACE_SHA256,
        "resolver_symbol": "collapse_semantic_transition_aliases",
        "lineage_symbol": "iter_candidate_lineage_from_selected_trace",
        "module_namespaces_disjoint": True,
    }
    equiv._install_semantic_lineage_finalizer(
        science_graph_trace=science,
        semantic_graph_trace=semantic,
        binding=binding,
        receipt_path=receipt,
        role="legacy",
        science_project_commit=equiv.SOURCE_COMMIT,
    )

    rows = list(
        science.iter_candidate_lineage_from_selected_trace({}, [], include_actions=False)
    )
    assert rows == [{"candidate_index": 0, "action_lineage_resolved": True}]
    value = equiv._validate_semantic_finalizer_receipt(
        receipt,
        role="legacy",
        science_project_commit=equiv.SOURCE_COMMIT,
    )
    assert value["patched_generation_symbols"] == []
    assert value["random_walk_patched"] is False
    assert value["candidate_universe_patched"] is False
    assert value["semantic_transition_alias_event_count"] == 1
    assert value["invocations"][0]["semantic_transition_count"] == 1


def test_checkpoint_prefix_semantic_finalizer_receipt_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    value = {
        "schema_version": equiv.SEMANTIC_FINALIZER_SCHEMA,
        "status": "PASS",
        "role": "legacy",
        "science_project_commit": equiv.SOURCE_COMMIT,
        "source_algorithm_commit": equiv.SOURCE_COMMIT,
        "patched_generation_symbols": [],
        "random_walk_patched": False,
        "candidate_generation_patched": False,
        "candidate_universe_patched": False,
        "delegation_scope": "post_walk_lineage_materialization_only",
        "all_lineage_rows_resolved": True,
        "semantic_finalizer_binding": {
            "status": "PASS",
            "commit": equiv.SEMANTIC_FINALIZER_COMMIT,
            "graph_trace_sha256": "0" * 64,
            "resolver_symbol": "collapse_semantic_transition_aliases",
            "lineage_symbol": "iter_candidate_lineage_from_selected_trace",
            "module_namespaces_disjoint": True,
        },
        "invocations": [
            {
                "status": "PASS",
                "semantic_transition_count": 1,
                "unresolved_row_count": 0,
            }
        ],
    }
    value["receipt_sha256"] = equiv._stable_json_sha256(value)
    _json(receipt, value)
    with pytest.raises(ValueError, match="semantic_finalizer_binding"):
        equiv._validate_semantic_finalizer_receipt(
            receipt,
            role="legacy",
            science_project_commit=equiv.SOURCE_COMMIT,
        )


def _traced_source(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = root / "trace-on"
    source.mkdir(parents=True)
    payload = _text(source / "counterfactuals.pt", "immutable-payload\n")
    monkeypatch.setattr(mut, "SOURCE_PAYLOAD_SHA256", sha256_file(payload))
    parent_ids = [f"parent-{index:04d}" for index in range(mut.SOURCE_PARENT_COUNT)]
    parent_sha = stable_json_sha256(parent_ids)
    monkeypatch.setattr(mut, "SOURCE_PARENT_ORDER_SHA256", parent_sha)
    config = {
        "dataset": "mutagenicity",
        "route": "project_adapted",
        "mode": "full",
        "project_commit": mut.SOURCE_PROJECT_COMMIT,
        "upstream_commit": mut.SOURCE_UPSTREAM_COMMIT,
        "parent_limit": mut.SOURCE_PARENT_COUNT,
        "generation_parent_ids": parent_ids,
        "generation_parent_ids_sha256": parent_sha,
        "parameters": dict(mut.SOURCE_PARAMETERS),
        "dataset_audit": {
            "dataset_fingerprint": mut.SOURCE_DATASET_SHA256,
            "generation_parent_ids_sha256": parent_sha,
        },
        "gnn": {"checkpoint_sha256": mut.SOURCE_GNN_SHA256},
        "distance_model": {"checkpoint_sha256": mut.SOURCE_DISTANCE_SHA256},
        "calibration_loaded": False,
        "test_loaded": False,
    }
    config_sha = stable_json_sha256(config)
    monkeypatch.setattr(mut, "SOURCE_CONFIG_SHA256", config_sha)
    config["config_sha256"] = config_sha
    _json(source / "resolved_config.json", config)
    _json(
        source / "run_manifest.json",
        {
            **config,
            "run_complete": True,
            "trace_enabled": True,
            "counterfactual_candidate_count": mut.SOURCE_CANDIDATE_COUNT,
            "counterfactuals_sha256": mut.SOURCE_PAYLOAD_SHA256,
        },
    )
    _json(
        source / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "counterfactuals_sha256": mut.SOURCE_PAYLOAD_SHA256,
        },
    )
    _json(
        source / "freeze_only_recovery.json",
        {"recovery_completed": True, "completed_steps": mut.SOURCE_STEPS},
    )
    _json(
        source / "frozen_payload_closure_audit.json",
        {"closure_complete": True, "post_write_reload_verified": True},
    )
    _json(source / "adoption_manifest.json", {"status": "PASS"})
    _json(
        source / "trace/trace_summary.json",
        {
            "trace_only": True,
            "candidate_count": mut.SOURCE_CANDIDATE_COUNT,
            "candidate_lineage_resolved_count": mut.SOURCE_CANDIDATE_COUNT,
            "algorithm_rerun": False,
        },
    )
    _json(source / "trace/_TRACE_COMPLETE.json", {"trace_complete": True})
    _json(source / "trace/candidate_action_lineage.json", {"rows": []})
    _text(source / "trace/candidate_action_lineage_index.jsonl", "{}\n")
    _json(source / "trace/selected_action_trace_manifest.json", {"status": "PASS"})
    proc = root / "proc"
    proc.mkdir()
    return source, proc


def test_traced_source_gate_recomputes_identity_and_writes_pass_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, proc = _traced_source(tmp_path, monkeypatch)
    output = tmp_path / "source-gate"
    result = mut.publish_traced_source_gate(
        source_root=source, output_dir=output, proc_root=proc
    )
    assert result["status"] == "PASS"
    assert (output / "source_gate.json").is_file()
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"

    config = json.loads((source / "resolved_config.json").read_text())
    config["generation_parent_ids"][0] = "changed"
    _json(source / "resolved_config.json", config)
    with pytest.raises(RepairManifestError, match="identity mismatch"):
        mut.verify_traced_source(
            source_root=source, proc_root=proc, hash_payload=False
        )


def test_traced_gate_failure_never_creates_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-source-gate"

    def _fail(**_kwargs: object) -> dict[str, object]:
        raise RepairManifestError("fault injection")

    monkeypatch.setattr(mut, "verify_traced_source", _fail)
    with pytest.raises(RepairManifestError, match="fault injection"):
        mut.publish_traced_source_gate(
            source_root=tmp_path, output_dir=output, proc_root=tmp_path
        )
    assert not output.exists()


def test_copied_payload_and_claimed_traceoff_manifest_are_not_execution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mut, "SOURCE_PARENT_COUNT", 2)
    parents = ["p0", "p1"]
    parent_sha = stable_json_sha256(parents)
    monkeypatch.setattr(mut, "SOURCE_PARENT_ORDER_SHA256", parent_sha)
    traced = tmp_path / "traced"
    reference = tmp_path / "reference"
    checkpoints = reference / "generation_checkpoints"
    mirror = tmp_path / "mirror"
    traced.mkdir()
    reference.mkdir()
    checkpoints.mkdir()
    mirror.mkdir()
    _text(traced / "counterfactuals.pt", "same bytes\n")
    payload = _text(reference / "counterfactuals.pt", "same bytes\n")
    base = {
        "dataset": "mutagenicity",
        "route": "project_adapted",
        "mode": "full",
        "project_commit": mut.INSTRUMENTATION_PROJECT_COMMIT,
        "upstream_commit": mut.SOURCE_UPSTREAM_COMMIT,
        "parent_limit": 2,
        "generation_parent_ids": parents,
        "generation_parent_ids_sha256": parent_sha,
        "parameters": dict(mut.SOURCE_PARAMETERS),
        "dataset_audit": {
            "dataset_fingerprint": mut.SOURCE_DATASET_SHA256,
            "generation_parent_ids_sha256": parent_sha,
        },
        "gnn": {"checkpoint_sha256": mut.SOURCE_GNN_SHA256},
        "distance_model": {"checkpoint_sha256": mut.SOURCE_DISTANCE_SHA256},
        "calibration_loaded": False,
        "test_loaded": False,
        "generation_checkpoint_root": str(checkpoints),
        "generation_checkpoint_mirror_root": str(mirror),
        "generation_resume_supported": True,
        "official_compatibility_patches": [],
    }
    config_sha = stable_json_sha256(base)
    _json(reference / "resolved_config.json", {**base, "config_sha256": config_sha})
    _json(
        reference / "run_manifest.json",
        {
            **base,
            "config_sha256": config_sha,
            "run_complete": True,
            "algorithm_rerun": False,
            "trace_enabled": False,
            "trace_summary": None,
            "trace_parity": None,
            "counterfactual_candidate_count": 1,
            "counterfactuals_sha256": sha256_file(payload),
        },
    )
    _json(
        reference / "_RUN_COMPLETE.json",
        {"run_complete": True, "counterfactuals_sha256": sha256_file(payload)},
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(RepairManifestError, match="algorithm_rerun"):
        mut.verify_traceoff_reference(
            reference_root=reference,
            traced_source_root=traced,
            expected_project_commit=mut.INSTRUMENTATION_PROJECT_COMMIT,
            expected_scientific_command_sha256="a" * 64,
            checkpoint_root=checkpoints,
            mirror_root=mirror,
            proc_root=proc,
        )


def test_resume_rejects_unproven_mirror_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "generation"
    output.mkdir()
    checkpoint_root = output / "generation_checkpoints"
    mirror = tmp_path / "mirror"
    step = mirror / "step-000000000500"
    step.mkdir(parents=True)
    _json(step / mut.MIRRORED_FILENAME, {})
    validation = SimpleNamespace(
        checkpoint_dir=step,
        completed_step=500,
        checkpoint_digest="d" * 64,
    )
    monkeypatch.setattr(mut, "list_generation_checkpoints", lambda _root: [step])
    monkeypatch.setattr(mut, "validate_generation_checkpoint", lambda _root: validation)
    with pytest.raises(RepairManifestError, match="mirror proof is invalid"):
        mut.prepare_generation_resume(
            output_root=output,
            checkpoint_root=checkpoint_root,
            mirror_root=mirror,
        )


def _builder_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="mut-parity-fixture-", dir="/private/tmp")
    root = Path(temporary.name)
    runtime = root / "runtime"
    (runtime / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs").mkdir(
        parents=True
    )
    (runtime / "locks").mkdir()
    control = runtime / "control"
    (control / mut.SOURCE_NAMESPACE / "manifests").mkdir(parents=True)
    proc = root / "proc"
    proc.mkdir()
    cgroup = root / "cgroup"
    cgroup.mkdir()
    flock = _text(root / "bin/flock", "#!/bin/bash\nexit 0\n")
    flock.chmod(0o755)
    source = root / "source"
    upstream = root / "upstream"
    dataset = root / "dataset"
    molclr = root / "molclr"
    legacy_science = root / "legacy-science"
    instrumented_science = root / "instrumented-science"
    for directory in (
        source,
        upstream,
        dataset,
        molclr,
        legacy_science,
        instrumented_science,
    ):
        directory.mkdir()
    gnn = _text(root / "gnn.pt")
    distance = _text(root / "distance.pt")
    monkeypatch.setattr(mut, "SOURCE_GNN_SHA256", sha256_file(gnn))
    monkeypatch.setattr(mut, "SOURCE_DISTANCE_SHA256", sha256_file(distance))
    repair_v2_manifest = _json(root / "repair-v2.json", {"fixture": True})
    repair_v2_root = root / "repair-v2-controller"
    repair_v2_root.mkdir()
    failed_mut = root / "failed-mut"
    common = failed_mut / "common_recourse"
    common.mkdir(parents=True)
    repair_v1_manifest = _json(root / "repair-v1.json", {"fixture": True})
    repair_v1_root = root / "repair-v1-controller"
    repair_v1_root.mkdir()
    threshold_output = root / "threshold"
    threshold_output.mkdir()
    threshold = _json(threshold_output / "frozen_threshold_contract.json", {"status": "PASS"})
    aids_manifest = _json(root / "aids-v4.json", {"fixture": True})
    aids_root = root / "aids-v4-controller"
    aids_root.mkdir()
    aids_sha = sha256_file(aids_manifest)
    standardized = {
        "dataset_csv": _text(root / "dataset.csv"),
        "teacher_path": _text(root / "teacher.pkl"),
        "molclr_root": molclr,
        "molclr_checkpoint": _text(root / "molclr.pt"),
    }
    head = mut._git_head(Path.cwd().resolve())
    monkeypatch.setattr(
        mut,
        "_git_head",
        lambda project: (
            mut.SOURCE_PROJECT_COMMIT
            if Path(project) == legacy_science
            else mut.INSTRUMENTATION_PROJECT_COMMIT
            if Path(project) == instrumented_science
            else head
        ),
    )
    monkeypatch.setattr(mut, "_git_is_ancestor", lambda **_kwargs: True)
    monkeypatch.setattr(mut, "_require_clean_tracked_worktree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mut,
        "LEGACY_SOURCE_INVENTORY_SHA256",
        mut.instrumentation_source_inventory(legacy_science)["inventory_sha256"],
    )
    monkeypatch.setattr(
        mut,
        "INSTRUMENTATION_SOURCE_INVENTORY_SHA256",
        mut.instrumentation_source_inventory(instrumented_science)["inventory_sha256"],
    )
    monkeypatch.setattr(
        mut,
        "verify_fix_ancestry",
        lambda **_kwargs: {
            "execution_head": head,
            "required_fix_commit": mut.VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
            "is_ancestor": "true",
        },
    )
    monkeypatch.setattr(
        mut,
        "verify_traced_source",
        lambda **_kwargs: {"status": "PASS", "test_loaded": False},
    )
    monkeypatch.setattr(
        mut,
        "verify_checkout",
        lambda *_args, **_kwargs: {
            "passed": True,
            "actual_commit": mut.SOURCE_UPSTREAM_COMMIT,
        },
    )
    monkeypatch.setattr(
        mut,
        "_repair_v2_manifest",
        lambda **_kwargs: {"status": "PASS", "task_id": mut.REPAIR_V2_MUT_TASK_ID},
    )
    monkeypatch.setattr(
        mut,
        "_validate_common_recourse_source",
        lambda **_kwargs: {
            "status": "PASS",
            "source_common_recourse_root": str(common),
        },
    )
    monkeypatch.setattr(
        mut,
        "verify_repair_v1_source",
        lambda **_kwargs: {"status": "PASS", "semantic": {"threshold_contract": str(threshold)}},
    )
    monkeypatch.setattr(
        mut,
        "_aids_manifest",
        lambda **_kwargs: SimpleNamespace(
            controller_id="four_methods_four_datasets_aids_comrecgc_repair_v4",
            sha256=aids_sha,
        ),
    )
    spec = _json(
        root / "spec.json",
        {
            "schema_version": mut.SPEC_SCHEMA,
            "controller_id": mut.CONTROLLER_ID,
            "paper_frozen": True,
            "run_tastemolnet": 0,
            "runtime_root": str(runtime),
            "control_root": str(control),
            "project_root": str(Path.cwd().resolve()),
            "python": str(Path(os.sys.executable).resolve()),
            "proc_root": str(proc),
            "cgroup_memory_root": str(cgroup),
            "min_cgroup_free_bytes": mut.AIDS_MINIMUM_HEADROOM_BYTES,
            "highmem_lock_path": str(runtime / "locks/comrecgc_common_recourse_highmem.lock"),
            "flock_bin": str(flock),
            "fresh_output_root": str(
                runtime
                / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut-v1"
            ),
            "verify_comrecgc_checkout_safe_git_fix_commit": mut.VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
            "traced_source_root": str(source),
            "instrumentation_equivalence": {
                "legacy_project_root": str(legacy_science),
                "instrumentation_project_root": str(instrumented_science),
                "steps": mut.INSTRUMENTATION_EQUIVALENCE_STEPS,
            },
            "repair_v2": {
                "manifest": str(repair_v2_manifest),
                "root": str(repair_v2_root),
                "failed_mut_output": str(failed_mut),
            },
            "repair_v1": {
                "manifest": str(repair_v1_manifest),
                "root": str(repair_v1_root),
                "mut_threshold_output": str(threshold_output),
            },
            "aids_dependency": {
                "controller_id": "four_methods_four_datasets_aids_comrecgc_repair_v4",
                "task_id": "aids_comrecgc_standardized_external_memory",
                "wrapper": "run_aids_comrecgc_repair_v4_supervisor.sh",
                "terminal_contract": "comrecgc_standardized_v1",
                "min_cgroup_free_bytes": mut.AIDS_V4_MINIMUM_HEADROOM_BYTES,
                "manifest": str(aids_manifest),
                "expected_manifest_sha256": aids_sha,
                "root": str(aids_root),
                "expected_output": str(root / "aids-output"),
            },
            "replay": {
                "upstream_root": str(upstream),
                "dataset_dir": str(dataset),
                "gnn_checkpoint": str(gnn),
                "distance_checkpoint": str(distance),
                "parent_limit": mut.SOURCE_PARENT_COUNT,
                "batch_size": mut.SOURCE_BATCH_SIZE,
                "parameters": dict(mut.SOURCE_PARAMETERS),
                "trace_enabled": False,
            },
            "standardization": {key: str(value) for key, value in standardized.items()},
        },
    )
    return temporary, spec


def _independent_v2_spec(spec_path: Path) -> dict[str, object]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runtime = Path(str(spec["runtime_root"]))
    spec.update(
        {
            "schema_version": mut.SPEC_SCHEMA_V2,
            "controller_id": mut.CONTROLLER_ID_V2,
            "remove_artificial_aids_wait": True,
            "independent_launch_authority": mut.INDEPENDENT_LAUNCH_AUTHORITY,
            "fresh_output_root": str(
                runtime
                / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1"
                / "repairs"
                / mut.CONTROLLER_ID_V2
            ),
            "min_cgroup_free_bytes": mut.MUT_FROZEN_MINIMUM_HEADROOM_BYTES,
        }
    )
    spec.pop("aids_dependency")
    _json(spec_path, spec)
    return spec


def test_builder_has_attempt_gates_exact_dependencies_and_cpu_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec = _builder_fixture(monkeypatch)
    try:
        payload, summary = mut.build_payload(spec_path=spec)
        result = mut.validate_payload(payload)
        tasks = {task["id"]: task for task in payload["tasks"]}
        assert summary["task_count"] == result["task_count"] == 8
        assert payload["runtime"]["max_transient_retries"] == 1
        assert payload["runtime"]["max_cpu_tasks"] == 1
        assert all("{attempt}" in str(task["expected_output"]) for task in tasks.values())
        generation = tasks[mut.TRACEOFF_TASK_ID]
        equivalence = tasks[mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID]
        assert equivalence["resource"] == "gpu"
        assert equivalence["gpu_lock_mode"] == "exclusive"
        assert equivalence["environment"]["GPU_REQUIRED"] == "1"
        assert mut.AIDS_WAIT_TASK_ID in equivalence["depends_on"]
        assert mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID in generation["depends_on"]
        assert generation["resource"] == "gpu"
        assert generation["gpu_lock_mode"] == "exclusive"
        assert generation["environment"]["GPU_REQUIRED"] == "1"
        assert generation["environment"]["MUT_GENERATION_OUTPUT"].endswith(
            "/traceoff-generation"
        )
        parity = tasks[mut.PARITY_TASK_ID]
        standard = tasks[mut.STANDARDIZE_TASK_ID]
        assert parity["environment"]["GPU_REQUIRED"] == "0"
        assert standard["environment"]["GPU_REQUIRED"] == "0"
        assert "{dep_mut_assert_trace_on_off_parity_output}" in standard["environment"][
            "MUT_PARITY_GATE"
        ]
        assert payload["mut_traceoff_parity_contract"][
            "aids_controller_id"
        ].endswith("repair_v4")
    finally:
        temporary.cleanup()


def test_v1_validator_accepts_pre_v2_contract_without_new_mode_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        payload, _summary = mut.build_payload(spec_path=spec_path)
        contract = payload["mut_traceoff_parity_contract"]
        assert "mut_independent_of_aids_final" not in contract
        assert "independent_launch_authority" not in contract
        result = mut.validate_payload(payload)
        assert result["controller_id"] == mut.CONTROLLER_ID
        assert result["task_count"] == 8
    finally:
        temporary.cleanup()


def test_independent_v2_has_seven_tasks_and_preserves_real_parity_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        _independent_v2_spec(spec_path)

        def _unexpected_aids_lookup(**_kwargs: object) -> object:
            pytest.fail("independent v2 must not inspect an AIDS controller")

        monkeypatch.setattr(mut, "_aids_manifest", _unexpected_aids_lookup)
        payload, summary = mut.build_payload(spec_path=spec_path)
        result = mut.validate_payload(payload)
        tasks = {task["id"]: task for task in payload["tasks"]}

        assert summary["controller_id"] == mut.CONTROLLER_ID_V2
        assert summary["task_count"] == result["task_count"] == 7
        assert set(tasks) == {
            mut.TRACE_SOURCE_TASK_ID,
            mut.THRESHOLD_TASK_ID,
            mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID,
            mut.TRACEOFF_TASK_ID,
            mut.PARITY_TASK_ID,
            mut.COMMON_TASK_ID,
            mut.STANDARDIZE_TASK_ID,
        }
        assert mut.AIDS_WAIT_TASK_ID not in tasks
        assert tasks[mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID]["depends_on"] == [
            mut.TRACE_SOURCE_TASK_ID
        ]
        assert tasks[mut.TRACEOFF_TASK_ID]["depends_on"] == [
            mut.TRACE_SOURCE_TASK_ID,
            mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID,
        ]

        contract = payload["mut_traceoff_parity_contract"]
        assert contract["schema_version"] == mut.SPEC_SCHEMA_V2
        assert contract["waits_for_reviewed_aids_repair_pass"] is False
        assert contract["mut_independent_of_aids_final"] is True
        assert (
            contract["independent_launch_authority"]
            == mut.INDEPENDENT_LAUNCH_AUTHORITY
        )
        assert contract["source_algorithm_commit"] == mut.SOURCE_PROJECT_COMMIT
        assert (
            contract["execution_instrumentation_commit"]
            == mut.INSTRUMENTATION_PROJECT_COMMIT
        )
        assert contract["instrumentation_equivalence_steps"] == 500
        assert contract["source_parameters"] == mut.SOURCE_PARAMETERS
        assert contract["source_trace_enabled"] is True
        assert contract["reference_trace_enabled"] is False
        assert contract["reference_self_comparison_forbidden"] is True
        assert contract["trace_fields_stripped"] is False
        assert "aids_controller_id" not in contract
        assert "aids_manifest_sha256" not in contract
        assert tasks[mut.TRACEOFF_TASK_ID]["environment"][
            "MUT_EXPECTED_SCIENTIFIC_COMMAND_SHA256"
        ] == contract["traceoff_scientific_command_sha256"]
    finally:
        temporary.cleanup()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda spec: spec.pop("independent_launch_authority"),
            "independent launch authority",
        ),
        (
            lambda spec: spec.__setitem__("remove_artificial_aids_wait", False),
            "remove_artificial_aids_wait=true",
        ),
        (
            lambda spec: spec.__setitem__("aids_dependency", {"forbidden": True}),
            "must omit aids_dependency",
        ),
        (
            lambda spec: spec.__setitem__(
                "min_cgroup_free_bytes", mut.AIDS_MINIMUM_HEADROOM_BYTES
            ),
            "frozen 440 GiB",
        ),
        (
            lambda spec: spec.__setitem__(
                "fresh_output_root",
                str(Path(str(spec["fresh_output_root"])).with_name("another-root")),
            ),
            "fresh output root must be exact",
        ),
    ],
)
def test_independent_v2_rejects_missing_authority_or_latent_aids_dependency(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        spec = _independent_v2_spec(spec_path)
        mutation(spec)
        _json(spec_path, spec)
        with pytest.raises(RepairManifestError, match=message):
            mut.build_payload(spec_path=spec_path)
    finally:
        temporary.cleanup()


def test_independent_v2_validator_rejects_reintroduced_aids_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        _independent_v2_spec(spec_path)
        payload, _summary = mut.build_payload(spec_path=spec_path)
        tasks = {task["id"]: task for task in payload["tasks"]}
        tasks[mut.INSTRUMENTATION_EQUIVALENCE_TASK_ID]["depends_on"].append(
            mut.AIDS_WAIT_TASK_ID
        )
        with pytest.raises(ControllerError, match="Unknown task dependencies"):
            mut.validate_payload(payload)
    finally:
        temporary.cleanup()


def test_independent_v2_builds_only_at_new_controller_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        spec = _independent_v2_spec(spec_path)
        expected = (
            Path(str(spec["control_root"]))
            / mut.SOURCE_NAMESPACE
            / "manifests"
            / f"{mut.CONTROLLER_ID_V2}.json"
        )
        result = mut.build_manifest(spec_path=spec_path, output_path=expected)
        assert result["controller_id"] == mut.CONTROLLER_ID_V2
        assert Path(str(result["manifest"])) == expected.resolve()
        assert expected.is_file()
    finally:
        temporary.cleanup()


def test_builder_v5_dependency_consumes_gate_and_freezes_attempt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary, spec_path = _builder_fixture(monkeypatch)
    try:
        spec = json.loads(spec_path.read_text())
        aids = spec["aids_dependency"]
        aids["controller_id"] = mut.AIDS_EXACT_ROUTE_V5_CONTROLLER_ID
        aids["task_id"] = mut.AIDS_EXACT_ROUTE_V5_TASK_ID
        aids["wrapper"] = "run_aids_comrecgc_exact_route_v5_supervisor.sh"
        aids.pop("expected_output")
        gate = Path(aids["root"]) / "tasks" / aids["task_id"] / "gate.json"
        gate.parent.mkdir(parents=True)
        _json(gate, {"status": "PASS"})
        output = spec_path.parent / "aids-v5-output/attempt-1"
        output.mkdir(parents=True)
        aids["task_gate"] = str(gate)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        gate_sha = sha256_file(gate)
        authority = {
            "status": "PASS",
            "controller_id": mut.AIDS_EXACT_ROUTE_V5_CONTROLLER_ID,
            "task_id": mut.AIDS_EXACT_ROUTE_V5_TASK_ID,
            "task_gate": str(gate.resolve()),
            "task_gate_sha256": gate_sha,
            "attempt": 1,
            "passing_output": str(output.resolve()),
        }
        monkeypatch.setattr(
            mut,
            "_aids_manifest",
            lambda **_kwargs: SimpleNamespace(
                controller_id=mut.AIDS_EXACT_ROUTE_V5_CONTROLLER_ID,
                sha256=aids["expected_manifest_sha256"],
            ),
        )
        monkeypatch.setattr(
            mut, "resolve_aids_passing_output", lambda **_kwargs: authority
        )
        payload, _summary = mut.build_payload(spec_path=spec_path)
        tasks = {task["id"]: task for task in payload["tasks"]}
        wait = tasks[mut.AIDS_WAIT_TASK_ID]
        assert wait["input_manifest"] == str(gate.resolve())
        assert "--task-gate" in wait["command"]
        assert gate_sha in wait["command"]
        contract = payload["mut_traceoff_parity_contract"]
        assert contract["aids_pass_authority"]["attempt"] == 1
        assert contract["aids_task_gate_sha256"] == gate_sha
    finally:
        temporary.cleanup()


def test_known_failed_aids_v3_and_manifest_replacement_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    namespace = control / mut.SOURCE_NAMESPACE
    manifests = namespace / "manifests"
    manifests.mkdir(parents=True)
    controller_id = "four_methods_four_datasets_aids_comrecgc_repair_v4"
    controller_root = namespace / controller_id
    controller_root.mkdir()
    manifest_path = _json(manifests / f"{controller_id}.json", {"fixture": True})
    monkeypatch.setattr(
        mut,
        "load_controller_manifest",
        lambda _path: SimpleNamespace(
            controller_id=controller_id,
            sha256="a" * 64,
            by_id={},
        ),
    )
    with pytest.raises(RepairManifestError, match="SHA256 changed"):
        mut._aids_manifest(
            source_manifest=manifest_path.resolve(),
            source_controller_root=controller_root.resolve(),
            control_root=control.resolve(),
            expected_controller_id=controller_id,
            expected_task_id="aids_comrecgc_standardized_cpu_bounded",
            expected_wrapper="run_comrecgc_standardized_continuation_cpu_bounded.sh",
            expected_manifest_sha256="b" * 64,
        )
    with pytest.raises(RepairManifestError, match="Known OOM-failed"):
        mut._aids_manifest(
            source_manifest=manifest_path.resolve(),
            source_controller_root=controller_root.resolve(),
            control_root=control.resolve(),
            expected_controller_id=mut.AIDS_CONTROLLER_ID,
            expected_task_id=mut.AIDS_TASK_ID,
            expected_wrapper="run_comrecgc_standardized_continuation_cpu_highmem.sh",
            expected_manifest_sha256="a" * 64,
        )


def test_aids_v5_pass_authority_resolves_attempt_one_and_hash_binds_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / mut.SOURCE_NAMESPACE
    manifests = namespace / "manifests"
    manifests.mkdir(parents=True)
    controller = namespace / mut.AIDS_EXACT_ROUTE_V5_CONTROLLER_ID
    task_root = controller / "tasks" / mut.AIDS_EXACT_ROUTE_V5_TASK_ID
    task_root.mkdir(parents=True)
    output_template = tmp_path / "outputs/attempt-{attempt}"
    output = tmp_path / "outputs/attempt-1"
    output.mkdir(parents=True)
    manifest_path = _json(manifests / "v5.json", {"fixture": True})
    manifest_sha = sha256_file(manifest_path)
    _json(
        task_root / "state.json",
        {
            "task_id": mut.AIDS_EXACT_ROUTE_V5_TASK_ID,
            "state": "PASS",
            "instances": {
                "main": {
                    "state": "PASS",
                    "run_id": "v5-run-attempt-1",
                    "attempt": 1,
                    "expected_output": str(output),
                }
            },
        },
    )
    gate_path = _json(
        task_root / "gate.json",
        {
            "task_id": mut.AIDS_EXACT_ROUTE_V5_TASK_ID,
            "status": "PASS",
            "runs": [
                {
                    "instance_id": "main",
                    "state": "PASS",
                    "run_id": "v5-run-attempt-1",
                    "attempt": 1,
                    "expected_output": str(output),
                }
            ],
        },
    )
    monkeypatch.setattr(
        mut,
        "load_controller_manifest",
        lambda _path: SimpleNamespace(
            controller_id=mut.AIDS_EXACT_ROUTE_V5_CONTROLLER_ID,
            sha256=manifest_sha,
            by_id={
                mut.AIDS_EXACT_ROUTE_V5_TASK_ID: SimpleNamespace(
                    expected_output=str(output_template)
                )
            },
        ),
    )
    monkeypatch.setattr(
        mut,
        "verify_controller_terminal",
        lambda **_kwargs: {"status": "PASS", "source_output_root": str(output)},
    )
    authority = mut.resolve_aids_passing_output(
        source_manifest=manifest_path,
        source_controller_root=controller,
        task_gate=gate_path,
        task_id=mut.AIDS_EXACT_ROUTE_V5_TASK_ID,
        expected_manifest_sha256=manifest_sha,
        expected_task_gate_sha256=sha256_file(gate_path),
        proc_root=tmp_path,
    )
    assert authority["attempt"] == 1
    assert authority["passing_output"] == str(output.resolve())
    assert authority["task_gate_sha256"] == sha256_file(gate_path)
    gate = json.loads(gate_path.read_text())
    gate["runs"][0]["expected_output"] = str(tmp_path / "tampered")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(RepairManifestError, match="SHA256 changed"):
        mut.resolve_aids_passing_output(
            source_manifest=manifest_path,
            source_controller_root=controller,
            task_gate=gate_path,
            task_id=mut.AIDS_EXACT_ROUTE_V5_TASK_ID,
            expected_manifest_sha256=manifest_sha,
            expected_task_gate_sha256=authority["task_gate_sha256"],
            proc_root=tmp_path,
        )


def test_wrapper_template_and_slurm_are_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts/autodl/run_mut_traceoff_stage_highmem.sh").read_text()
    assert "--trace-output-dir" not in wrapper
    assert "--parity-reference" not in wrapper
    for token in (
        "--resume",
        "--checkpoint-mirror-root",
        "--expected-project-commit",
        "COMRECGC_HIGHMEM_LOCK_PATH",
        "GPU_REQUIRED",
        "MUT_STAGE_OUTPUT",
        "MUT_SEMANTIC_FINALIZER_PROJECT_ROOT",
        "--semantic-finalizer-project-root",
        "final-five-closeout-582bc4b-20260902T040000Z",
    ):
        assert token in wrapper
    template = json.loads(
        (root / "configs/autodl/mut_traceoff_parity_v1.template.json").read_text()
    )
    assert template["replay"]["trace_enabled"] is False
    assert template["replay"]["parameters"] == mut.SOURCE_PARAMETERS
    assert template["instrumentation_equivalence"]["steps"] == 500
    assert template["aids_dependency"]["controller_id"].endswith("repair_v4")
    assert template["aids_dependency"]["expected_manifest_sha256"].startswith("__FILL")
    independent = json.loads(
        (
            root
            / "configs/autodl/mut_traceoff_parity_v2_independent.template.json"
        ).read_text()
    )
    assert independent["schema_version"] == mut.SPEC_SCHEMA_V2
    assert independent["controller_id"] == mut.CONTROLLER_ID_V2
    assert independent["remove_artificial_aids_wait"] is True
    assert (
        independent["independent_launch_authority"]
        == mut.INDEPENDENT_LAUNCH_AUTHORITY
    )
    assert "aids_dependency" not in independent
    assert independent["replay"] == template["replay"]
    assert independent["instrumentation_equivalence"] == template[
        "instrumentation_equivalence"
    ]
    assert independent["min_cgroup_free_bytes"] == template[
        "min_cgroup_free_bytes"
    ]
    assert (
        independent["min_cgroup_free_bytes"]
        == mut.MUT_FROZEN_MINIMUM_HEADROOM_BYTES
    )
    for key in (
        "python",
        "proc_root",
        "cgroup_memory_root",
        "highmem_lock_path",
        "flock_bin",
        "verify_comrecgc_checkout_safe_git_fix_commit",
        "traced_source_root",
        "repair_v2",
        "repair_v1",
        "standardization",
    ):
        assert independent[key] == template[key]
    for relative in (
        "scripts/slurm/manage_mut_traceoff_parity_v1.sh",
        "scripts/slurm/run_mut_comrecgc_parity_standardization.sh",
        "scripts/slurm/run_mut_traceoff_stage_highmem.sh",
        "scripts/slurm/run_mut_checkpoint_instrumentation_equivalence.sh",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "do not submit",
        ):
            assert token in text
    slurm = (root / "scripts/slurm/manage_mut_traceoff_parity_v1.sh").read_text()
    assert "mut_traceoff_parity_v2_independent.template.json" in slurm


def test_manage_cli_routes_threshold_and_future_aids_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def _threshold(**kwargs: object) -> dict[str, object]:
        captured["threshold"] = kwargs
        return {"status": "PASS"}

    def _wait(**kwargs: object) -> dict[str, object]:
        captured["wait"] = kwargs
        return {"status": "PASS"}

    monkeypatch.setattr(manage, "publish_threshold_source_gate", _threshold)
    monkeypatch.setattr(manage, "wait_for_aids_pass", _wait)
    common = [
        "--source-manifest",
        str(tmp_path / "manifest.json"),
        "--source-controller-root",
        str(tmp_path / "controller"),
        "--control-root",
        str(tmp_path / "control"),
        "--expected-output-root",
        str(tmp_path / "expected"),
    ]
    assert (
        manage.main(
            [
                "verify-threshold",
                *common,
                "--project-root",
                str(tmp_path / "project"),
                "--output-dir",
                str(tmp_path / "threshold-gate"),
            ]
        )
        == 0
    )
    assert "expected_controller_id" not in captured["threshold"]
    assert (
        manage.main(
            [
                "wait-aids",
                "--expected-controller-id",
                "future-v4",
                "--expected-task-id",
                "aids-task",
                "--expected-wrapper",
                "wrapper.sh",
                "--expected-manifest-sha256",
                "a" * 64,
                *common,
                "--output-dir",
                str(tmp_path / "aids-gate"),
            ]
        )
        == 0
    )
    assert captured["wait"]["expected_controller_id"] == "future-v4"
    assert captured["wait"]["expected_task_id"] == "aids-task"
    assert captured["wait"]["expected_wrapper"] == "wrapper.sh"
    assert captured["wait"]["expected_manifest_sha256"] == "a" * 64


def test_instrumentation_source_delta_is_explicit_and_fail_closed() -> None:
    files: dict[str, object] = {}
    for relative in equiv.SOURCE_FILES:
        if relative in {
            "src/baselines/comrecgc/generation_checkpoint.py",
            "src/baselines/comrecgc/generation_loop.py",
        }:
            files[relative] = {"present": False}
        else:
            files[relative] = {
                "present": True,
                "sha256": "a" * 64,
                "top_level_definition_ast_sha256": {},
            }
    legacy = {"files": files}
    instrumented_files = json.loads(json.dumps(files))
    for relative in (
        "src/baselines/comrecgc/generation_checkpoint.py",
        "src/baselines/comrecgc/generation_loop.py",
    ):
        instrumented_files[relative] = {
            "present": True,
            "sha256": "b" * 64,
            "top_level_definition_ast_sha256": {"checkpoint": "b" * 64},
        }
    # Every reviewed changed module must contain one allowlisted AST delta.
    allowlisted = {
        "scripts/baselines/comrecgc/run_generation.py": "main",
        "src/baselines/comrecgc/runtime.py": "run_project_generation",
        "src/baselines/comrecgc/graph_trace.py": "ActionTraceRecorder",
        "src/baselines/comrecgc/live_graph_state.py": "LiveGraphState",
        "src/baselines/comrecgc/transition_cache.py": "CompactMoveScopedTransitionMap",
        "src/baselines/comrecgc/storage_guard.py": "StorageGuard",
    }
    for relative, definition in allowlisted.items():
        files[relative]["top_level_definition_ast_sha256"] = {definition: "a" * 64}
        instrumented_files[relative]["top_level_definition_ast_sha256"] = {
            definition: "b" * 64
        }
        instrumented_files[relative]["sha256"] = "b" * 64
    result = equiv._source_delta_audit(legacy, {"files": instrumented_files})
    assert result["status"] == "PASS"
    instrumented_files["src/baselines/comrecgc/model_adapter.py"]["sha256"] = "c" * 64
    failed = equiv._source_delta_audit(legacy, {"files": instrumented_files})
    assert failed["status"] == "FAIL"
    assert any("model_adapter.py" in value for value in failed["failures"])


def test_instrumentation_gate_requires_resume_payload_rng_and_pass_last(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gate"
    root.mkdir()
    gate = {
        "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
        "status": "PASS",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": 500,
        "seed": 0,
        "source_algorithm_commit": mut.SOURCE_PROJECT_COMMIT,
        "execution_instrumentation_commit": mut.INSTRUMENTATION_PROJECT_COMMIT,
        "step_action_trace_exact": True,
        "rng_state_exact": True,
        "checkpoint_mirror_verified": True,
        "checkpoint_resume_exercised": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "payload_equivalence": {
            "failures": [],
            "candidate_parity": {"trace_parity_passed": True},
        },
        "source_audit": {
            "legacy": {
                "project_commit": mut.SOURCE_PROJECT_COMMIT,
                "inventory_sha256": mut.LEGACY_SOURCE_INVENTORY_SHA256,
            },
            "instrumented": {
                "project_commit": mut.INSTRUMENTATION_PROJECT_COMMIT,
                "inventory_sha256": mut.INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
            },
            "delta_audit": {"status": "PASS", "failures": []},
        },
        "failures": [],
    }
    gate["summary_sha256"] = stable_json_sha256(gate)
    path = _json(root / "equivalence.json", gate)
    _text(root / "PASS", "PASS\n")
    result = mut.validate_instrumentation_equivalence_gate(
        gate_path=path,
        expected_legacy_inventory_sha256=mut.LEGACY_SOURCE_INVENTORY_SHA256,
        expected_instrumentation_inventory_sha256=(
            mut.INSTRUMENTATION_SOURCE_INVENTORY_SHA256
        ),
    )
    assert result["status"] == "PASS"
    gate["checkpoint_resume_exercised"] = False
    gate["summary_sha256"] = stable_json_sha256(
        {key: value for key, value in gate.items() if key != "summary_sha256"}
    )
    _json(path, gate)
    with pytest.raises(RepairManifestError, match="checkpoint_resume_exercised"):
        mut.validate_instrumentation_equivalence_gate(
            gate_path=path,
            expected_legacy_inventory_sha256=mut.LEGACY_SOURCE_INVENTORY_SHA256,
            expected_instrumentation_inventory_sha256=(
                mut.INSTRUMENTATION_SOURCE_INVENTORY_SHA256
            ),
        )


def test_instrumentation_diagnostic_exercises_real_sigterm_boundary(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "instrumented-checkpoint-mirror"
    marker = mirror / "step-000000000250/_CHECKPOINT_MIRRORED.json"
    script = _text(
        tmp_path / "emit_checkpoint.py",
        "from pathlib import Path\n"
        "import time\n"
        f"path = Path({str(marker)!r})\n"
        "path.parent.mkdir(parents=True)\n"
        "path.write_text('{}\\n', encoding='utf-8')\n"
        "time.sleep(30)\n",
    )
    (tmp_path / "instrumented").mkdir()
    proof = tmp_path / "instrumented-interruption-proof.json"
    equiv._interrupt_at_checkpoint(
        [os.sys.executable, str(script)],
        log=tmp_path / "instrumented.log",
        environment=os.environ,
        mirror_root=mirror,
        proof_path=proof,
    )
    payload = json.loads(proof.read_text(encoding="utf-8"))
    assert payload["signal"] == "SIGTERM"
    assert payload["completed_checkpoint_step"] == 250
    assert payload["run_complete_absent_after_interrupt"] is True
