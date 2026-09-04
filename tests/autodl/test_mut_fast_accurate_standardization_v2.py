from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.run_comrecgc_standardized_continuation import ContinuationInputs
from scripts.autodl import run_mut_comrecgc_parity_standardization as standardizer
from src.baselines.comrecgc.contracts import (
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    sha256_file,
    stable_json_sha256,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _historical_adoption(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "historical-generation"
    source.mkdir()
    (source / "counterfactuals.pt").write_bytes(b"historical trace-on fixture\n")
    lineage = source / "trace/candidate_action_lineage.json"
    _json(lineage, {"status": "PASS", "candidate_count": standardizer.SOURCE_CANDIDATE_COUNT})

    equivalence = tmp_path / "evidence/equivalence.json"
    equivalence_payload = {
        "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
        "status": "PASS",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": 500,
        "seed": 0,
        "source_algorithm_commit": standardizer.SOURCE_PROJECT_COMMIT,
        "execution_instrumentation_commit": standardizer.INSTRUMENTATION_PROJECT_COMMIT,
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
                "project_commit": standardizer.SOURCE_PROJECT_COMMIT,
                "inventory_sha256": standardizer.LEGACY_SOURCE_INVENTORY_SHA256,
            },
            "instrumented": {
                "project_commit": standardizer.INSTRUMENTATION_PROJECT_COMMIT,
                "inventory_sha256": (
                    standardizer.INSTRUMENTATION_SOURCE_INVENTORY_SHA256
                ),
            },
            "delta_audit": {"status": "PASS", "failures": []},
        },
        "failures": [],
    }
    equivalence_payload["summary_sha256"] = stable_json_sha256(equivalence_payload)
    _json(equivalence, equivalence_payload)
    (equivalence.parent / "PASS").write_text("PASS\n", encoding="utf-8")

    common = tmp_path / "historical-common"
    external = common / "external"
    vectors = external / "vectors.npy"
    vectors.parent.mkdir(parents=True)
    vectors.write_bytes(b"vectors\n")
    vectors_sha = sha256_file(vectors)
    pair_manifest = external / "pair-store.json"
    _json(
        pair_manifest,
        {
            "run_complete": True,
            "vectors_path": str(vectors.resolve()),
            "vectors_sha256": vectors_sha,
            "scientific_identity": {
                "dataset": "mutagenicity",
                "counterfactuals_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
                "candidate_graph_hashes_sha256": "b" * 64,
                "generation_indices_sha256": "c" * 64,
            },
        },
    )
    dbscan_manifest = external / "dbscan.json"
    _json(
        dbscan_manifest,
        {
            "run_complete": True,
            "approximation_used": False,
            "scientific_identity": {
                "vectors_path": str(vectors.resolve()),
                "vectors_sha256": vectors_sha,
            },
        },
    )
    common_manifest = common / "run_manifest.json"
    _json(
        common_manifest,
        {
            "dataset": "mutagenicity",
            "method": METHOD,
            "run_complete": True,
            "source_counterfactuals_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
            "common_recourse_count": 19,
            "external_memory_artifacts": {
                "engine": "external_memory_exact_v1",
                "pair_store_manifest": str(pair_manifest.resolve()),
                "pair_store_manifest_sha256": sha256_file(pair_manifest),
                "dbscan_manifest": str(dbscan_manifest.resolve()),
                "dbscan_manifest_sha256": sha256_file(dbscan_manifest),
            },
        },
    )

    universe = "b" * 64
    candidate_binding = tmp_path / "evidence/candidate_universe_binding.json"
    candidate_binding_payload = {
        "schema_version": "mut_candidate_pair_dbscan_binding_v1",
        "status": "PASS",
        "binding_kind": standardizer.TRANSITIVE_BINDING_KIND,
        "source_native_candidate_universe_sha": universe,
        "pair_store_source_candidate_universe_sha": universe,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": universe,
        "dbscan_approximation_used": False,
    }
    candidate_binding_payload["binding_sha256"] = stable_json_sha256(
        candidate_binding_payload
    )
    _json(candidate_binding, candidate_binding_payload)
    receipt = tmp_path / "historical-adoption.json"
    receipt_payload = {
            "schema_version": standardizer.HISTORICAL_ADOPTION_SCHEMA,
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": METHOD,
            "historical_artifact_adopted": True,
            "historical_source_trace_enabled": True,
            "traceoff_reference_rerun": False,
            "trace_parity_passed": False,
            "500_step_semantic_equivalence_passed": True,
            "adoption_without_full_50k_parity_rerun_authorized": True,
            "generation_complete": True,
            "generation_steps": 50_000,
            "M_MAX": 50_000,
            "M_EFFECTIVE": 50_000,
            "candidate_capacity": 100_000,
            "candidate_count": standardizer.SOURCE_CANDIDATE_COUNT,
            "lineage_pass": True,
            "candidate_freeze_pass": True,
            "checkpoint_reload_pass": True,
            "no_test_leakage": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "no_active_writer": True,
            "pair_store_reused": True,
            "dbscan_reused": True,
            "pair_store_recompute_performed": False,
            "dbscan_recompute_performed": False,
            "candidate_universe_sha": universe,
            "source_native_candidate_universe_sha": universe,
            "pair_store_source_candidate_universe_sha": universe,
            "dbscan_native_candidate_universe_sha": None,
            "dbscan_transitively_bound_candidate_universe_sha": universe,
            "candidate_universe_binding_state": "PASS",
            "transitive_binding_kind": standardizer.TRANSITIVE_BINDING_KIND,
            "pair_candidate_graph_hashes_sha256": universe,
            "dbscan_native_candidate_universe_field_present": False,
            "dbscan_universe_binding_via_pair_vectors": True,
            "candidate_pair_dbscan_binding_path": str(candidate_binding.resolve()),
            "candidate_pair_dbscan_binding_file_sha256": sha256_file(
                candidate_binding
            ),
            "source_generation_root": str(source.resolve()),
            "source_payload_path": str((source / "counterfactuals.pt").resolve()),
            "source_payload_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
            "source_lineage_path": str(lineage.resolve()),
            "source_lineage_sha256": sha256_file(lineage),
            "500_step_semantic_equivalence_receipt_path": str(equivalence.resolve()),
            "500_step_semantic_equivalence_receipt_sha256": sha256_file(equivalence),
            "source_common_recourse_root": str(common.resolve()),
            "source_common_recourse_manifest_path": str(common_manifest.resolve()),
            "source_common_recourse_manifest_sha256": sha256_file(common_manifest),
            "source_pair_store_manifest_path": str(pair_manifest.resolve()),
            "source_pair_store_manifest_sha256": sha256_file(pair_manifest),
            "source_dbscan_manifest_path": str(dbscan_manifest.resolve()),
            "source_dbscan_manifest_sha256": sha256_file(dbscan_manifest),
            "common_recourse_count": 19,
    }
    receipt_payload["binding_sha256"] = stable_json_sha256(receipt_payload)
    _json(receipt, receipt_payload)
    return receipt, source


def _inputs(tmp_path: Path, *, source: Path, output: Path) -> ContinuationInputs:
    directories = {
        name: tmp_path / name for name in ("upstream", "dataset", "molclr")
    }
    for path in directories.values():
        path.mkdir(exist_ok=True)
    files = {
        name: tmp_path / name
        for name in (
            "distance.pt",
            "dataset.csv",
            "teacher.pkl",
            "molclr.pt",
            "thresholds.json",
        )
    }
    for path in files.values():
        path.write_bytes(b"fixture\n")
    return ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=source,
        upstream_root=directories["upstream"],
        dataset_dir=directories["dataset"],
        source_csv=None,
        distance_checkpoint=files["distance.pt"],
        dataset_csv=files["dataset.csv"],
        teacher_path=files["teacher.pkl"],
        molclr_root=directories["molclr"],
        molclr_checkpoint=files["molclr.pt"],
        thresholds_path=files["thresholds.json"],
        output_root=output,
        device="cpu",
        theta_star=None,
        cost_cap=None,
    )


def test_historical_adoption_reopens_trace_on_50k_transitive_binding(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)

    reopened = standardizer._validate_historical_adoption(receipt, source_root=source)

    assert reopened["historical_source_trace_enabled"] is True
    assert reopened["traceoff_reference_rerun"] is False
    assert reopened["trace_parity_passed"] is False
    assert reopened["generation_steps"] == 50_000
    assert reopened["M_EFFECTIVE"] == 50_000
    assert reopened["transitive_binding_kind"] == standardizer.TRANSITIVE_BINDING_KIND
    assert reopened["candidate_universe_sha"] == "b" * 64
    assert reopened["source_native_candidate_universe_sha"] == "b" * 64
    assert reopened["pair_store_source_candidate_universe_sha"] == "b" * 64
    assert reopened["dbscan_native_candidate_universe_sha"] is None
    assert reopened["dbscan_transitively_bound_candidate_universe_sha"] == "b" * 64
    assert reopened["pair_candidate_graph_hashes_sha256"] == "b" * 64
    assert reopened["dbscan_native_candidate_universe_field_present"] is False


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("historical_source_trace_enabled", False),
        ("traceoff_reference_rerun", True),
        ("trace_parity_passed", True),
        ("500_step_semantic_equivalence_passed", False),
        ("adoption_without_full_50k_parity_rerun_authorized", False),
        ("pair_store_source_candidate_universe_sha", "e" * 64),
    ),
)
def test_historical_adoption_rejects_false_or_drifted_claims(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = bad_value
    _json(receipt, payload)

    with pytest.raises(ValueError, match="Historical Mut adoption is invalid"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_receipt_universe_not_in_pair_identity(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    for field in (
        "candidate_universe_sha",
        "source_native_candidate_universe_sha",
        "pair_store_source_candidate_universe_sha",
        "dbscan_transitively_bound_candidate_universe_sha",
        "pair_candidate_graph_hashes_sha256",
    ):
        payload[field] = "e" * 64
    _json(receipt, payload)

    with pytest.raises(ValueError, match="pair_store_candidate_universe_binding"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_minimal_forged_equivalence_receipt(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    equivalence = Path(payload["500_step_semantic_equivalence_receipt_path"])
    _json(
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
    payload["500_step_semantic_equivalence_receipt_sha256"] = sha256_file(
        equivalence
    )
    payload["binding_sha256"] = stable_json_sha256(
        {key: item for key, item in payload.items() if key != "binding_sha256"}
    )
    _json(receipt, payload)

    with pytest.raises(ValueError, match="historical_generation_evidence"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_binding_digest_drift(tmp_path: Path) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["common_recourse_count"] += 1
    _json(receipt, payload)

    with pytest.raises(ValueError, match="binding_sha256"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_run_emits_truthful_fast_accurate_v2_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    inputs = _inputs(tmp_path, source=source, output=tmp_path / "fresh-output")
    monkeypatch.setattr(
        standardizer,
        "validate_adopted_generation",
        lambda _inputs: {
            "source_generation_root": str(source.resolve()),
            "counterfactual_candidate_count": standardizer.SOURCE_CANDIDATE_COUNT,
        },
    )
    monkeypatch.setattr(standardizer, "verify_checkout", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(standardizer, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        standardizer,
        "_verify_adopted_generation_integrity",
        lambda _adoption: {"status": "PASS"},
    )

    def _no_science_commands(
        current: ContinuationInputs, **_kwargs: object
    ) -> list[tuple[str, list[str], Path, str]]:
        teacher_sha = sha256_file(current.teacher_path)
        _json(
            current.output_root / "standardized/run_manifest.json",
            {
                "dataset_key": "mutagenicity",
                "cf_mode": CF_MODE,
                "distance_line": DISTANCE_LINE,
                "teacher_sha256": teacher_sha,
            },
        )
        _json(
            current.output_root / "standardized/freeze_manifest.json",
            {"dataset_key": "mutagenicity"},
        )
        return []

    monkeypatch.setattr(standardizer, "_commands", _no_science_commands)

    result = standardizer.run(
        inputs,
        common_adoption_path=None,
        historical_adoption_path=receipt,
    )

    assert result["schema_version"] == standardizer.FAST_ACCURATE_RUN_SCHEMA
    assert result["historical_source_trace_enabled"] is True
    assert result["traceoff_reference_rerun"] is False
    assert result["trace_parity_passed"] is False
    assert result["500_step_semantic_equivalence_passed"] is True
    assert result["adoption_without_full_50k_parity_rerun_authorized"] is True
    assert result["generation_steps"] == result["M_EFFECTIVE"] == 50_000
    assert result["trace_parity_path"] is None
    assert result["trace_parity_sha256"] is None
    assert result["candidate_universe_sha"] == "b" * 64
    assert result["pair_candidate_graph_hashes_sha256"] == "b" * 64
    assert result["dbscan_native_candidate_universe_field_present"] is False
    assert result["dbscan_universe_binding_via_pair_vectors"] is True
    assert (inputs.output_root / "PASS").read_bytes() == b"PASS\n"


def test_historical_adoption_reopens_current_same_contract_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.utils.autodl_mut_first_divergence_v1 import stable_sha256
    from src.utils.autodl_mut_post_ab_continuation_v1 import (
        EXECUTION_COMMIT,
        SOURCE_COMMIT,
        UPSTREAM_COMMIT,
    )
    from src.utils.autodl_mut_same_contract_ab_v1 import (
        build_same_contract_ab_spec,
    )

    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    legacy_gate = Path(payload["500_step_semantic_equivalence_receipt_path"])
    old_gate = legacy_gate.parent / "trace_on_off_500_step_equivalence.json"
    payload["500_step_semantic_equivalence_receipt_path"] = str(old_gate)
    input_manifest: dict[str, object] = {
        "schema_version": "mut_trace_equivalence_input_manifest_v1",
        "dataset": "mutagenicity",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "upstream_commit": UPSTREAM_COMMIT,
        "formal_M_MAX": 50_000,
        "comparison_steps": 500,
        "post_reload_steps": 10,
        "candidate_capacity": 100_000,
        "seed": 0,
        "parent_limit": 1448,
        "batch_size": 128,
        "device": "cuda:0",
        "arms_sequential": True,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "pythonhashseed": "0",
    }
    input_manifest["manifest_sha256"] = stable_sha256(input_manifest)
    input_path = old_gate.parent / "equivalence_input_manifest.json"
    _json(input_path, input_manifest)
    exact_fields = {
        key: True
        for key in (
            "trace_on_off_stepwise_exact",
            "step_action_trace_exact",
            "rng_state_exact",
            "classifier_probability_trace_exact",
            "step_semantic_fields_present",
            "trace_on_checkpoint_reload_pass",
            "trace_off_checkpoint_reload_pass",
            "post_reload_trace_mode_equivalence_pass",
            "step500_checkpoint_serialized_candidate_records_exact",
            "step500_checkpoint_candidate_universe_exact",
            "checkpoint_algorithm_scientific_state_exact",
            "checkpoint_rng_state_exact",
            "checkpoint_sqlite_logical_state_exact",
            "checkpoint_graph_registry_exact",
            "resolved_config_scientific_binding_exact",
        )
    }
    gate: dict[str, object] = {
        "schema_version": "mut_trace_on_off_500_step_equivalence_v1",
        "status": "PASS",
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "source_algorithm_commit": SOURCE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "formal_M_MAX": 50_000,
        "steps_compared": 500,
        "post_reload_steps_compared": 10,
        "trace_on_trace_enabled": True,
        "trace_off_trace_enabled": False,
        "trace_only_files_excluded_from_scientific_digest": True,
        "post_walk_prefix_finalization_performed": False,
        "full_50k_trace_on_off_parity_claimed": False,
        "arms_overlapped": False,
        "max_concurrent_arms": 1,
        "calibration_loaded": False,
        "test_loaded": False,
        "first_semantic_divergence_step": None,
        "failures": [],
        "trace_on_observer_log_audit": {"status": "PASS"},
        "trace_off_observer_log_audit": {"status": "PASS"},
        "input_manifest": str(input_path),
        "input_manifest_sha256": sha256_file(input_path),
        **exact_fields,
    }
    gate["summary_sha256"] = stable_sha256(gate)
    _json(old_gate, gate)

    control = tmp_path / "ab-control"
    run_root = tmp_path / "ab-run"
    ab_spec = build_same_contract_ab_spec(
        {
            "task_id": "mut-ab",
            "attempt_uuid": "12345678-1234-4234-9234-123456789abc",
            "controller_project_root": str(tmp_path / "controller"),
            "controller_commit": "a" * 40,
            "python": str(tmp_path / "python"),
            "runner_path": str(tmp_path / "controller/scripts/autodl/run_mut_trace_mode_equivalence.py"),
            "legacy_project_root": str(tmp_path / "legacy"),
            "execution_project_root": str(tmp_path / "execution"),
            "historical_artifact_root": str(source),
            "upstream_root": str(tmp_path / "upstream"),
            "dataset_dir": str(tmp_path / "dataset"),
            "gnn_checkpoint": str(tmp_path / "gnn.pt"),
            "distance_checkpoint": str(tmp_path / "distance.pt"),
            "rf_oracle": str(tmp_path / "rf.pkl"),
            "run_root": str(run_root),
            "output_dir": str(old_gate.parent),
            "control_root": str(control),
            "lease_path": str(control / "owner.lease"),
            "gpu_lock_root": str(tmp_path / "locks"),
            "gpu_uuid": "GPU-test-uuid",
            "gpu_index": 0,
        },
        check_files=False,
    )
    ab_spec_path = tmp_path / "ab-task-spec.json"
    _json(ab_spec_path, ab_spec)
    owner_terminal = control / "terminal.json"
    _json(
        owner_terminal,
        {
            "schema_version": "mut_same_contract_ab_owner_terminal_v1",
            "task_id": "mut-ab",
            "status": "PASS_TRACE_MODE_EQUIVALENCE",
            "fresh_50k_started": False,
            "equivalence_gate": str(old_gate),
            "equivalence_gate_sha256": sha256_file(old_gate),
        },
    )
    authorization = tmp_path / "authorization.json"
    _json(authorization, {"controller_id": "authorized-controller"})
    monkeypatch.setattr(
        standardizer,
        "validate_authorization_receipt",
        lambda *_args, **_kwargs: ({"authorization_sha256": "d" * 64}, sha256_file(authorization)),
    )
    payload.update(
        {
            "same_contract_ab_spec_path": str(ab_spec_path),
            "same_contract_ab_spec_sha256": sha256_file(ab_spec_path),
            "same_contract_ab_owner_terminal_path": str(owner_terminal),
            "same_contract_ab_owner_terminal_sha256": sha256_file(owner_terminal),
            "same_contract_gate_path": str(old_gate),
            "same_contract_gate_sha256": sha256_file(old_gate),
            "same_contract_gate_summary_sha256": gate["summary_sha256"],
            "500_step_semantic_equivalence_receipt_sha256": sha256_file(old_gate),
            "trace_on_adoption_authorization_path": str(authorization),
            "trace_on_adoption_authorization_file_sha256": sha256_file(authorization),
            "trace_on_adoption_authorization_sha256": "d" * 64,
        }
    )
    payload["binding_sha256"] = stable_json_sha256(
        {key: item for key, item in payload.items() if key != "binding_sha256"}
    )
    _json(receipt, payload)
    reopened = standardizer._validate_historical_adoption(receipt, source_root=source)
    assert reopened["same_contract_gate_sha256"] == sha256_file(old_gate)
