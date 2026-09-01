from __future__ import annotations

import json
import pickle
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from src.baselines.comrecgc.audit import validate_final_manifest, validate_monotonic
from src.baselines.comrecgc.contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    DISTANCE_LINE,
    GenerationParameters,
    RecourseParameters,
    ContractError,
    UPSTREAM_COMMIT,
    ordered_ids_sha256,
    sha256_file,
    write_json,
)
from src.baselines.comrecgc.project_dataset import project_label_to_internal
from src.baselines.comrecgc.preregistration import validate_chemistry_trace_evidence
from src.baselines.comrecgc.runtime import (
    ACTIVE_MOVE_TRANSITION_PATCH,
    _EndpointSafeGraphMap,
    _MoveScopedTransitionMap,
    _materialize_dataset_indices,
    patched_official_runtime,
    validate_counterfactual_payload,
)
from src.baselines.comrecgc import upstream


def test_generation_profiles_are_frozen() -> None:
    smoke = GenerationParameters.for_mode("smoke")
    assert smoke.steps == 100
    assert smoke.sample_size == 128
    smoke.validate("smoke")
    GenerationParameters.for_mode("full").validate("full")
    invalid = GenerationParameters.for_mode("smoke")
    with pytest.raises(ContractError):
        invalid.validate("full")


def test_common_recourse_profiles_are_frozen() -> None:
    assert RecourseParameters.for_mode("smoke").recourse_size == 5
    assert RecourseParameters.for_mode("full").cf_size == 100_000
    RecourseParameters.for_mode("full").validate("full")


def test_order_hash_is_order_sensitive() -> None:
    assert ordered_ids_sha256(["a", "b"]) != ordered_ids_sha256(["b", "a"])


def test_atomic_json_write(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_mutagenicity_chemistry_requires_true_trace_parity(tmp_path: Path) -> None:
    evidence = tmp_path / "trace_summary.json"
    write_json(
        evidence,
        {
            "trace_only": True,
            "rng_calls_added": 0,
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(tmp_path / "_TRACE_COMPLETE.json", {"trace_complete": True})

    with pytest.raises(ValueError, match="trace parity"):
        validate_chemistry_trace_evidence(evidence, dataset="mutagenicity")


def _historical_mut_adoption_evidence(tmp_path: Path) -> Path:
    lineage = tmp_path / "candidate_action_lineage.json"
    write_json(lineage, {"candidate_count": 2, "lineage_pass": True})
    equivalence = tmp_path / "equivalence.json"
    write_json(
        equivalence,
        {
            "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
            "status": "PASS",
            "paper_eligible": False,
            "dataset": "mutagenicity",
            "steps": 500,
            "step_action_trace_exact": True,
            "rng_state_exact": True,
            "checkpoint_mirror_verified": True,
            "checkpoint_resume_exercised": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    evidence = tmp_path / "historical_adoption.json"
    write_json(
        evidence,
        {
            "schema_version": "mut_comrecgc_historical50k_adoption_v2",
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": "COMRECGC",
            "historical_artifact_adopted": True,
            "historical_source_trace_enabled": True,
            "traceoff_reference_rerun": False,
            "trace_parity_passed": False,
            "500_step_semantic_equivalence_passed": True,
            "adoption_without_full_50k_parity_rerun_authorized": True,
            "generation_complete": True,
            "generation_steps": 50_000,
            "M_EFFECTIVE": 50_000,
            "candidate_capacity": 100_000,
            "candidate_count": 2,
            "lineage_pass": True,
            "candidate_freeze_pass": True,
            "checkpoint_reload_pass": True,
            "no_test_leakage": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "source_lineage_path": str(lineage.resolve()),
            "source_lineage_sha256": sha256_file(lineage),
            "500_step_semantic_equivalence_receipt_path": str(
                equivalence.resolve()
            ),
            "500_step_semantic_equivalence_receipt_sha256": sha256_file(
                equivalence
            ),
        },
    )
    return evidence


def test_mutagenicity_chemistry_accepts_truthful_historical_trace_on_adoption(
    tmp_path: Path,
) -> None:
    evidence = _historical_mut_adoption_evidence(tmp_path)

    result = validate_chemistry_trace_evidence(evidence, dataset="mutagenicity")

    assert result["trace_integrity_passed"] is True
    assert result["trace_parity_required"] is False
    assert result["trace_parity_passed"] is False
    assert result["historical_source_trace_enabled"] is True
    assert result["traceoff_reference_rerun"] is False
    assert result["500_step_semantic_equivalence_passed"] is True
    assert result["adoption_without_full_50k_parity_rerun_authorized"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("historical_source_trace_enabled", False),
        ("traceoff_reference_rerun", True),
        ("trace_parity_passed", True),
        ("500_step_semantic_equivalence_passed", False),
        ("adoption_without_full_50k_parity_rerun_authorized", False),
    ),
)
def test_mutagenicity_historical_adoption_rejects_false_provenance_claims(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    evidence = _historical_mut_adoption_evidence(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload[field] = value
    write_json(evidence, payload)

    with pytest.raises(ValueError, match=field):
        validate_chemistry_trace_evidence(evidence, dataset="mutagenicity")


def test_aids_chemistry_accepts_complete_streamed_trace_without_claiming_parity(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "trace_summary.json"
    write_json(
        evidence,
        {
            "trace_only": True,
            "rng_calls_added": 0,
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(tmp_path / "_TRACE_COMPLETE.json", {"trace_complete": True})

    result = validate_chemistry_trace_evidence(evidence, dataset="aids")

    assert result["trace_integrity_passed"] is True
    assert result["trace_parity_required"] is False
    assert result["trace_parity_passed"] is False
    assert result["trace_rng_evidence_kind"] == "trace_summary_explicit_zero"
    assert result["rng_calls_added"] == 0


def _write_freeze_only_trace_without_redundant_rng_field(tmp_path: Path) -> Path:
    generation = tmp_path / "generation"
    trace = generation / "trace"
    trace.mkdir(parents=True)
    lineage_path = trace / "candidate_action_lineage.json"
    selected_path = trace / "selected_action_trace_manifest.json"
    write_json(
        lineage_path,
        {
            "schema_version": 2,
            "format": "selected_trace_predecessor_index",
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(
        selected_path,
        {"schema_version": 1, "format": "chunked_jsonl", "row_count": 0},
    )
    lineage_audit = {
        "legacy_inference_invocation_count": 0,
        "legacy_missing_action_count": 0,
        "missing_action_fallback_count": 0,
        "recorded_action_replay_failed_count": 0,
        "recorded_action_replay_mismatch_count": 0,
    }
    evidence = trace / "trace_summary.json"
    summary = {
        "trace_schema_version": 1,
        "trace_only": True,
        "algorithm_rerun": False,
        "lineage_recovery_policy": "authoritative_backing_freeze_only_v3",
        "candidate_lineage_format": "selected_trace_predecessor_index",
        "candidate_count": 2,
        "candidate_lineage_resolved_count": 2,
        "candidate_lineage_path": str(lineage_path.resolve()),
        "selected_trace_path": str(selected_path.resolve()),
        "lineage_recovery_audit": lineage_audit,
        "frozen_payload_closure": {
            "schema_version": "comrecgc_frozen_payload_closure_v7",
            "closure_complete": True,
            "scientific_parameters_changed": False,
            "candidate_order_changed": False,
            "candidate_payload_changed": False,
            "post_write_reload_verified": True,
            "original_trace_hash_roundtrip_verified": True,
            "sha_mismatch_count": 0,
            "unresolved_hash_count": 0,
        },
    }
    write_json(evidence, summary)
    write_json(
        trace / "_TRACE_COMPLETE.json",
        {
            "trace_complete": True,
            "freeze_only_recovery": True,
            "candidate_lineage_sha256": sha256_file(lineage_path),
            "selected_trace_manifest_sha256": sha256_file(selected_path),
        },
    )
    counterfactuals_sha256 = "a" * 64
    write_json(
        generation / "run_manifest.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "algorithm_rerun": False,
            "counterfactual_candidate_count": 2,
            "counterfactuals_sha256": counterfactuals_sha256,
            "trace_summary": summary,
        },
    )
    recovery = {
        "schema_version": "comrecgc_completed_generation_freeze_audit_v4",
        "FREEZE_ONLY_RECOVERY_SAFE": True,
        "recovery_completed": True,
        "algorithm_rerun": False,
        "random_walk_complete": True,
        "rng_state_required_for_freeze_only": False,
        "rng_state_reason": (
            "Random walk is complete; freeze-only performs no proposal or RNG call."
        ),
        "checks": {"random_walk_complete": True, "closure_complete": True},
        "candidate_count": 2,
        "candidate_lineage_resolved_count": 2,
        "output_dir": str(generation.resolve()),
        "counterfactuals_sha256": counterfactuals_sha256,
    }
    recovery_path = generation / "freeze_only_recovery.json"
    write_json(recovery_path, recovery)
    write_json(
        generation / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "freeze_only_recovery": True,
            "counterfactuals_sha256": counterfactuals_sha256,
            "recovery_manifest_sha256": sha256_file(recovery_path),
        },
    )
    return evidence


def test_aids_chemistry_accepts_hash_closed_freeze_only_rng_omission(
    tmp_path: Path,
) -> None:
    evidence = _write_freeze_only_trace_without_redundant_rng_field(tmp_path)

    result = validate_chemistry_trace_evidence(evidence, dataset="aids")

    assert result["trace_integrity_passed"] is True
    assert result["trace_parity_passed"] is False
    assert result["rng_calls_added"] == 0
    assert (
        result["trace_rng_evidence_kind"]
        == "completed_walk_freeze_only_v3_v4_receipt"
    )
    assert result["freeze_only_recovery_sha256"] == sha256_file(
        tmp_path / "generation/freeze_only_recovery.json"
    )


def test_aids_chemistry_rejects_unscoped_missing_rng_field(tmp_path: Path) -> None:
    evidence = tmp_path / "trace_summary.json"
    write_json(
        evidence,
        {
            "trace_only": True,
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(tmp_path / "_TRACE_COMPLETE.json", {"trace_complete": True})

    with pytest.raises(ValueError, match="trace integrity"):
        validate_chemistry_trace_evidence(evidence, dataset="aids")


@pytest.mark.parametrize(
    "corruption",
    (
        "recovery_reason",
        "recovery_check",
        "terminal_recovery_hash",
        "lineage_marker_hash",
        "payload_closure",
    ),
)
def test_aids_chemistry_freeze_only_rng_compatibility_fails_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    evidence = _write_freeze_only_trace_without_redundant_rng_field(tmp_path)
    generation = tmp_path / "generation"
    recovery_path = generation / "freeze_only_recovery.json"
    terminal_path = generation / "_RUN_COMPLETE.json"
    marker_path = generation / "trace/_TRACE_COMPLETE.json"

    if corruption in {"recovery_reason", "recovery_check"}:
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        if corruption == "recovery_reason":
            recovery["rng_state_reason"] = "assumed"
        else:
            recovery["checks"]["closure_complete"] = False
        write_json(recovery_path, recovery)
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["recovery_manifest_sha256"] = sha256_file(recovery_path)
        write_json(terminal_path, terminal)
    elif corruption == "terminal_recovery_hash":
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["recovery_manifest_sha256"] = "0" * 64
        write_json(terminal_path, terminal)
    elif corruption == "lineage_marker_hash":
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["candidate_lineage_sha256"] = "0" * 64
        write_json(marker_path, marker)
    else:
        summary = json.loads(evidence.read_text(encoding="utf-8"))
        summary["frozen_payload_closure"]["closure_complete"] = False
        write_json(evidence, summary)
        run_manifest_path = generation / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["trace_summary"] = summary
        write_json(run_manifest_path, run_manifest)

    with pytest.raises(ValueError, match="trace integrity"):
        validate_chemistry_trace_evidence(evidence, dataset="aids")


def test_monotonicity_gate() -> None:
    validate_monotonic([0.0, 0.2, 0.2, 1.0], field="coverage")
    with pytest.raises(ContractError):
        validate_monotonic([0.0, 0.3, 0.2], field="coverage")


def test_final_semantic_gate() -> None:
    validate_final_manifest(
        {
            "method": "COMRECGC",
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "adaptation_mode": ADAPTATION_MODE,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        }
    )


def test_project_label_mapping_is_explicit() -> None:
    assert project_label_to_internal(1) == 0
    assert project_label_to_internal(0) == 1
    with pytest.raises(ContractError):
        project_label_to_internal(2)


def test_upstream_payload_contract() -> None:
    graph_map, candidates = validate_counterfactual_payload(
        {"graph_map": {"hash": [object()]}, "counterfactual_candidates": [{"graph_hash": "hash"}]}
    )
    assert list(graph_map) == ["hash"]
    assert candidates[0]["graph_hash"] == "hash"
    with pytest.raises(RuntimeError):
        validate_counterfactual_payload({"graph_map": {}, "counterfactual_candidates": []})


def test_upstream_git_scopes_dubious_owner_override_to_exact_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "migrated-read-only-checkout"
    root.mkdir()
    (root / ".git").mkdir()
    for filename in ("comrecgc.py", "common_recourse.py", "data.py", "gnn.py"):
        (root / filename).write_text(f"# {filename}\n", encoding="utf-8")

    expected_prefix = ["git", "-C", str(root.resolve())]
    observed: list[list[str]] = []
    temporary_configs: list[Path] = []
    original_global_config = upstream.os.environ.get("GIT_CONFIG_GLOBAL")

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        observed.append(list(argv))
        environment = kwargs.pop("env")
        assert isinstance(environment, dict)
        safe_config = Path(str(environment["GIT_CONFIG_GLOBAL"]))
        temporary_configs.append(safe_config)
        expected_config = f'[safe]\n\tdirectory = "{root.resolve()}"\n'
        if (
            list(argv[:3]) != expected_prefix
            or safe_config.read_text(encoding="utf-8") != expected_config
            or environment["GIT_CONFIG_NOSYSTEM"] != "1"
        ):
            raise upstream.subprocess.CalledProcessError(
                128,
                argv,
                stderr="fatal: detected dubious ownership in repository",
            )
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        stdout = UPSTREAM_COMMIT if argv[-2:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=f"{stdout}\n")

    monkeypatch.setattr(upstream.subprocess, "run", fake_run)

    assert upstream.validate_upstream_checkout(root) == root.resolve()
    assert observed == [
        [*expected_prefix, "rev-parse", "HEAD"],
        [*expected_prefix, "status", "--porcelain", "--untracked-files=all"],
    ]
    assert all("--global" not in argv for argv in observed)
    assert all(not path.exists() for path in temporary_configs)
    assert upstream.os.environ.get("GIT_CONFIG_GLOBAL") == original_global_config


def test_native_source_rows_are_eagerly_materialized() -> None:
    class LazyRows:
        def __init__(self) -> None:
            self.open = True

        def __getitem__(self, index: int) -> str:
            if not self.open:
                raise FileNotFoundError("relative processed path unavailable")
            return f"graph-{index}"

    rows = LazyRows()
    materialized = _materialize_dataset_indices(rows, [3, 1])
    rows.open = False

    assert materialized == ["graph-3", "graph-1"]


def test_native_runtime_freezes_feature_dimension_before_cwd_switch() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/baselines/comrecgc/runtime.py"
    ).read_text(encoding="utf-8")
    feature_line = source.index("num_features = int(graphs.num_features)")
    cwd_line = source.index("os.chdir(runtime_root)", feature_line)
    dataset_line = source.index("GraphListDataset(sources, num_features)", cwd_line)
    assert feature_line < cwd_line < dataset_line


def test_native_aids_gnn_does_not_reopen_the_trusted_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import types

    import src.baselines.comrecgc.runtime as runtime

    checkpoint = tmp_path / "data/aids/gnn/model_best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    observed: dict[str, object] = {}

    class Model:
        def __init__(self, **kwargs) -> None:
            observed["kwargs"] = kwargs

        def to(self, device: str):
            observed["device"] = device
            return self

        def load_state_dict(self, state_dict) -> None:
            observed["state_dict"] = state_dict

        def eval(self):
            observed["eval"] = True
            return self

    fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: {"weight": 1})
    monkeypatch.setattr(runtime, "_torch_stack", lambda: (fake_torch, object()))
    module = types.SimpleNamespace(GNN=Model)

    result = runtime._load_native_aids_gnn_from_trusted_features(
        gnn_module=module,
        upstream_root=tmp_path,
        num_features=9,
        device="cuda:0",
    )

    assert isinstance(result, Model)
    assert observed["kwargs"] == {
        "num_features": 9,
        "num_classes": 2,
        "num_layers": 3,
        "dim": 20,
        "dropout": 0.0,
    }
    assert observed["device"] == "cuda:0"
    assert observed["state_dict"] == {"weight": 1}
    assert observed["eval"] is True


def test_native_trusted_payload_is_resolved_before_upstream_chdir() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/baselines/comrecgc/runtime.py"
    ).read_text(encoding="utf-8")
    resolve_line = source.index("trusted_payload_path.resolve(strict=True)")
    cwd_line = source.index("os.chdir(Path(upstream_root)", resolve_line)
    load_line = source.index("load_aids_tensor_payload(\n                        trusted_payload_path", cwd_line)
    assert resolve_line < cwd_line < load_line


def test_endpoint_safe_graph_map_preserves_normal_deletion_and_serializes_plain_dict() -> None:
    module = type("Module", (), {})()
    module.counterfactual_candidates = [{"graph_hash": "tail"}]
    module.graph_index_map = {}
    graph_map = _EndpointSafeGraphMap(module, {"tail": [1], "keep": [2]})

    del graph_map["tail"]

    assert graph_map == {"keep": [2]}
    assert graph_map.missing_unmaterialized_eviction_count == 0
    restored = pickle.loads(pickle.dumps(graph_map))
    assert type(restored) is dict
    assert restored == {"keep": [2]}


def test_endpoint_safe_graph_map_only_allows_unmaterialized_tail_eviction() -> None:
    module = type("Module", (), {})()
    module.counterfactual_candidates = [{"graph_hash": "unmaterialized"}]
    module.graph_index_map = {}
    graph_map = _EndpointSafeGraphMap(module, {"keep": [2]})

    del graph_map["unmaterialized"]

    assert graph_map == {"keep": [2]}
    assert graph_map.missing_unmaterialized_eviction_count == 1
    with pytest.raises(KeyError):
        del graph_map["different"]
    module.graph_index_map["unmaterialized"] = 0
    with pytest.raises(KeyError):
        del graph_map["unmaterialized"]


def test_active_head_transition_eviction_is_deferred_for_1000_moves() -> None:
    module = SimpleNamespace(graph_index_map={}, graph_map={})
    transitions = _MoveScopedTransitionMap(module, {}, seed=0)
    checkpoint: bytes | None = None

    for step in range(1, 1_001):
        heads = (f"lead-{step}", f"follower-{step}")
        for index, graph_hash in enumerate(heads):
            module.graph_index_map[graph_hash] = index
            module.graph_map[graph_hash] = [f"graph-{graph_hash}"]
            transitions[graph_hash] = (f"transition-{graph_hash}",)

        transitions.begin_move(heads)
        module.graph_index_map.pop(heads[1])
        module.graph_map.pop(heads[1])
        del transitions[heads[1]]

        assert all(graph_hash in transitions for graph_hash in heads)
        assert transitions[heads[1]] == (f"transition-{heads[1]}",)
        if step == 500:
            checkpoint = pickle.dumps(transitions)
            restored = pickle.loads(checkpoint)
            assert type(restored) is dict
            assert restored == dict(transitions)

        transitions.end_move()
        assert heads[1] not in transitions
        module.graph_index_map.pop(heads[0])
        module.graph_map.pop(heads[0])
        del transitions[heads[0]]

    assert checkpoint is not None
    assert transitions.audit() == {
        "patch": ACTIVE_MOVE_TRANSITION_PATCH,
        "policy": "defer_current_head_eviction_until_move_complete",
        "move_count": 1_000,
        "deferred_deletion_count": 1_000,
        "applied_deferred_deletion_count": 1_000,
        "cancelled_deferred_deletion_count": 0,
        "missing_lookup_count": 0,
        "max_transition_size": 2,
        "rng_calls_added": 0,
        "candidate_order_changed": False,
        "scientific_parameters_changed": False,
    }


def test_transition_lookup_failure_reports_move_context() -> None:
    module = SimpleNamespace(graph_index_map={"missing": 0}, graph_map={"missing": [1]})
    transitions = _MoveScopedTransitionMap(module, {}, seed=13)
    transitions.begin_move(["lead", "missing"])

    with pytest.raises(RuntimeError) as error:
        transitions["missing"]

    message = str(error.value)
    assert "[COMRECGC_TRANSITION_STATE_ERROR]" in message
    assert "current_step=1" in message
    assert "head=1" in message
    assert "seed=13" in message
    assert "graph_hash=missing" in message
    assert "transition_size=0" in message
    assert "cache_size=1" in message
    transitions.end_move()


def test_runtime_defers_active_transition_cleanup_until_original_move_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.baselines.comrecgc.runtime as runtime

    module = SimpleNamespace(
        call=object(),
        neighbor_graph_access=lambda graph, _action: graph,
        graph_map={"lead": [1], "follower": [2]},
        graph_index_map={"lead": 0, "follower": 1},
        counterfactual_candidates=[],
        transitions={"lead": ("lead-transition",), "follower": ("follower-transition",)},
    )
    observed: dict[str, object] = {}

    def original_move(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        module.graph_index_map.pop("follower")
        module.graph_map.pop("follower")
        del module.transitions["follower"]
        observed["available_during_move"] = module.transitions["follower"]
        return (["lead", "lead"], False, None, None, None)

    module.move_to_next_graph = original_move
    monkeypatch.setattr(runtime, "_safe_call_factory", lambda **_kwargs: object())
    audit: dict[str, object] = {}

    def trace_wrap(original: object, traced_module: object) -> object:
        def traced(*args: object, **kwargs: object) -> tuple[object, ...]:
            result = original(*args, **kwargs)
            observed["available_to_trace"] = traced_module.transitions["follower"]
            return result

        return traced

    trace_recorder = SimpleNamespace(wrap_move=trace_wrap)

    with patched_official_runtime(
        module,
        model=object(),
        embedding_model=object(),
        gnn_device="cpu",
        embedding_device="cpu",
        batch_size=1,
        trace_recorder=trace_recorder,
        compatibility_audit=audit,
        preserve_active_transitions=True,
        seed=0,
    ):
        module.move_to_next_graph(
            graphs_hash=["lead", "follower"],
            start_graphs_hash=["lead", "follower"],
            importance_args={},
            teleport_probability=0.1,
        )
        assert "follower" not in module.transitions

    assert observed["available_during_move"] == ("follower-transition",)
    assert observed["available_to_trace"] == ("follower-transition",)
    assert type(module.transitions) is dict
    assert audit["transition_state"] == {
        "patch": ACTIVE_MOVE_TRANSITION_PATCH,
        "policy": "defer_current_head_eviction_until_move_complete",
        "move_count": 1,
        "deferred_deletion_count": 1,
        "applied_deferred_deletion_count": 1,
        "cancelled_deferred_deletion_count": 0,
        "missing_lookup_count": 0,
        "max_transition_size": 2,
        "rng_calls_added": 0,
        "candidate_order_changed": False,
        "scientific_parameters_changed": False,
    }


def test_runtime_installs_and_restores_compact_full_transition_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.baselines.comrecgc.runtime as runtime

    source = SimpleNamespace(value=10)
    module = SimpleNamespace(
        call=object(),
        neighbor_graph_access=lambda graph, action: SimpleNamespace(
            value=graph.value + int(action[1])
        ),
        graph_map={"source": [source, None, None]},
        graph_index_map={"source": 0},
        counterfactual_candidates=[],
        transitions={},
    )

    def move(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        target = module.neighbor_graph_access(source, ("ADD", 1))
        module.transitions["source"] = (
            ["target"],
            [target],
            [[0.7, 1.0]],
            [[1.0, 2.0]],
        )
        module.graph_map["target"] = [target, None, None]
        module.graph_index_map["target"] = 1
        assert module.transitions["source"][1][0].value == 11
        return (["target"], False, None, None, None)

    module.move_to_next_graph = move
    monkeypatch.setattr(runtime, "_safe_call_factory", lambda **_kwargs: object())
    monkeypatch.setattr(
        runtime,
        "_apply_neighbor_with_lineage",
        lambda original, graph, action: original(graph, action),
    )
    audit: dict[str, object] = {}

    with patched_official_runtime(
        module,
        model=object(),
        embedding_model=object(),
        gnn_device="cpu",
        embedding_device="cpu",
        batch_size=1,
        compatibility_audit=audit,
        preserve_active_transitions=True,
        compact_transitions=True,
        transition_expanded_capacity=1,
        seed=0,
    ):
        module.move_to_next_graph(
            graphs_hash=["source"],
            start_graphs_hash=["source"],
            importance_args={},
            teleport_probability=0.1,
        )

    assert module.transitions == {}
    transition_audit = audit["transition_state"]
    assert transition_audit["policy"] == "exact_action_replay_with_bounded_expanded_lru"
    assert transition_audit["expanded_capacity"] == 1
    assert transition_audit["model_recomputation_count"] == 0


def test_runtime_exception_cleanup_skips_full_live_graph_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.baselines.comrecgc.runtime as runtime

    original_call = object()
    original_neighbor = lambda graph, action: graph
    original_move = object()
    source_graph = SimpleNamespace(
        x=[[1.0]],
        edge_index=[[], []],
        num_nodes=1,
    )
    module = SimpleNamespace(
        call=original_call,
        neighbor_graph_access=original_neighbor,
        move_to_next_graph=original_move,
        graph_map={"source": [source_graph]},
        graph_index_map={"source": 0},
        counterfactual_candidates=[],
        transitions={},
    )
    monkeypatch.setattr(runtime, "_safe_call_factory", lambda **_kwargs: object())

    def forbidden_full_audit(_self: object) -> dict[str, object]:
        raise AssertionError("exception cleanup must not reconstruct transition graphs")

    monkeypatch.setattr(runtime.LiveGraphState, "audit", forbidden_full_audit)
    audit: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="fail-closed stop"):
        with patched_official_runtime(
            module,
            model=object(),
            embedding_model=object(),
            gnn_device="cpu",
            embedding_device="cpu",
            batch_size=1,
            compatibility_audit=audit,
            preserve_active_transitions=True,
            compact_transitions=True,
            graph_state_dir=tmp_path,
            seed=0,
        ):
            raise RuntimeError("fail-closed stop")

    graph_audit = audit["live_graph_state"]
    assert graph_audit["audit_scope"] == "exception_cleanup_runtime_diagnostics"
    assert graph_audit["integrity_audit_complete"] is False
    assert graph_audit["hot_cache_size"] == 1
    assert graph_audit["backing_store_size"] == 0
    assert type(module.graph_map) is dict
    assert module.graph_map == {"source": [source_graph]}
    assert module.transitions == {}
    assert module.call is original_call
    assert module.neighbor_graph_access is original_neighbor
    assert module.move_to_next_graph is original_move
    assert not hasattr(module, "comrecgc_live_graph_state")


@pytest.mark.parametrize("raise_inside", [False, True])
def test_endpoint_safe_runtime_restores_plain_map_and_official_functions(
    monkeypatch, raise_inside: bool
) -> None:
    import src.baselines.comrecgc.runtime as runtime

    original_call = object()
    original_neighbor = lambda graph, action: graph
    original_move = object()
    module = SimpleNamespace(
        call=original_call,
        neighbor_graph_access=original_neighbor,
        move_to_next_graph=original_move,
        graph_map={"keep": [1]},
        graph_index_map={},
        counterfactual_candidates=[{"graph_hash": "unmaterialized"}],
    )
    patched_call = object()
    monkeypatch.setattr(runtime, "_safe_call_factory", lambda **_kwargs: patched_call)
    audit: dict[str, object] = {}

    def exercise() -> None:
        with patched_official_runtime(
            module,
            model=object(),
            embedding_model=object(),
            gnn_device="cpu",
            embedding_device="cpu",
            batch_size=1,
            compatibility_audit=audit,
        ):
            assert module.call is patched_call
            del module.graph_map["unmaterialized"]
            if raise_inside:
                raise RuntimeError("expected")

    if raise_inside:
        with pytest.raises(RuntimeError, match="expected"):
            exercise()
    else:
        exercise()

    assert type(module.graph_map) is dict
    assert module.graph_map == {"keep": [1]}
    assert module.call is original_call
    assert module.neighbor_graph_access is original_neighbor
    assert module.move_to_next_graph is original_move
    assert audit == {
        "patch": "candidate_map_unmaterialized_eviction_none_safe_v1",
        "missing_unmaterialized_eviction_count": 1,
        "rng_calls_added": 0,
        "candidate_order_changed": False,
    }


def test_upstream_import_does_not_write_bytecode(tmp_path: Path, monkeypatch) -> None:
    observed: list[bool] = []
    original = sys.dont_write_bytecode
    monkeypatch.setattr(upstream, "validate_upstream_checkout", lambda path: tmp_path)

    def fake_import(name: str) -> ModuleType:
        observed.append(sys.dont_write_bytecode)
        return ModuleType(name)

    monkeypatch.setattr(upstream.importlib, "import_module", fake_import)
    with upstream.imported_upstream(tmp_path) as modules:
        assert set(modules) == set(upstream.UPSTREAM_MODULES)
        assert sys.dont_write_bytecode is True

    assert observed == [True] * len(upstream.UPSTREAM_MODULES)
    assert sys.dont_write_bytecode is original


def test_upstream_import_restores_bytecode_flag_after_error(
    tmp_path: Path, monkeypatch
) -> None:
    original = sys.dont_write_bytecode
    monkeypatch.setattr(upstream, "validate_upstream_checkout", lambda path: tmp_path)
    monkeypatch.setattr(
        upstream.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("import failed")),
    )

    with pytest.raises(RuntimeError, match="import failed"):
        with upstream.imported_upstream(tmp_path):
            pass

    assert sys.dont_write_bytecode is original
