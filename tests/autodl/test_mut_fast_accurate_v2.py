from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.utils import autodl_mut_fast_accurate_v2 as mut


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _stable_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _real_equivalence_receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
        "status": "PASS",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": 500,
        "seed": 0,
        "source_algorithm_commit": "1" * 40,
        "execution_instrumentation_commit": "2" * 40,
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
                "project_commit": "1" * 40,
                "inventory_sha256": SHA_A,
            },
            "instrumented": {
                "project_commit": "2" * 40,
                "inventory_sha256": SHA_B,
            },
            "delta_audit": {"status": "PASS", "failures": []},
        },
        "failures": [],
    }
    value["summary_sha256"] = _stable_sha(value)
    return value


def _historical_manifest() -> dict[str, object]:
    lineage = {
        "schema_version": "comrecgc_recorded_action_first_v3",
        "candidate_lineage_resolved_count": 100_235,
        "recorded_action_replay_mismatch_count": 0,
        "predecessor_unverified_conflict_count": 0,
        "predecessor_unresolved_legacy_conflict_count": 0,
        "predecessor_selected_parent_mismatch_count": 0,
        "selected_event_source_parent_mismatch_count": 0,
        # These are different units in the historical authority: selected
        # events versus unique converged predecessor graphs.
        "selected_event_target_parent_mismatch_count": 14,
        "predecessor_cross_parent_convergence_count": 1,
    }
    closure = {
        "closure_complete": True,
        "post_write_reload_verified": True,
        "candidate_order_changed": False,
        "candidate_payload_changed": False,
        "scientific_parameters_changed": False,
        "sha_mismatch_count": 0,
        "unresolved_hash_count": 0,
        "canonical_hash_algorithm": "sha256_stable_json_untyped_graph_v1",
        "graph_serialization_version": "normalized_untyped_graph_payload_v1",
    }
    return {
        "dataset": "mutagenicity",
        "method": "COMRECGC",
        "parameters": {
            "steps": 50_000,
            "candidate_capacity": 100_000,
            "seed": 0,
        },
        "run_complete": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "trace_enabled": True,
        "counterfactual_candidate_count": 100_235,
        "source_dataset_fingerprint": SHA_A,
        "generation_parent_ids_sha256": SHA_B,
        "gnn": {"checkpoint_sha256": SHA_C},
        "config_sha256": SHA_D,
        "cf_mode": "strict_flip",
        "lineage_recovery_audit": lineage,
        "trace_summary": {"frozen_payload_closure": closure},
    }


def _binding_evidence() -> tuple[dict[str, object], ...]:
    generation = {
        "run_complete": True,
        "counterfactuals_sha256": SHA_A,
        "manifest_sha256": SHA_B,
        "counterfactual_candidate_count": 100_235,
    }
    pair = {
        "run_complete": True,
        "scientific_identity": {
            "counterfactuals_sha256": SHA_A,
            "generation_manifest_sha256": SHA_B,
            "candidate_count": 50_620,
            "candidate_graph_hashes_sha256": SHA_C,
        },
        "recourse_vectors_sha256": SHA_D,
    }
    dbscan = {
        "run_complete": True,
        "scientific_identity": {
            "vectors_sha256": SHA_D,
            "vectors_shape": [813_595, 64],
        },
    }
    return generation, pair, dbscan


def _convergence_window(step: int) -> dict[str, object]:
    return {
        "step": step,
        "checkpoint_committed": True,
        "evidence_split": "train",
        "top100_candidate_jaccard": 0.995,
        "top20_provisional_rule_jaccard": 0.96,
        "candidate_frequency_rank_spearman": 0.995,
        "absolute_train_coverage_gain": 0.004,
        "lineage_error_count": 0,
        "valid_unique_candidate_count": 10,
        "checkpoint_reload_pass": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def test_parse_complete_cgroup_v2_and_keep_missing_fields_explicit() -> None:
    files = {
        "memory.max": str(480 * mut.GIB),
        "memory.high": "max\n",
        "memory.current": str(300 * mut.GIB),
        "memory.peak": None,
        "memory.stat": "anon 10\nfile 20\ninactive_file 7\nslab_reclaimable 3\n",
        "memory.events": "low 0\nhigh 2\nmax 0\noom 0\noom_kill 0\n",
        "memory.events.local": "low 0\nhigh 1\nmax 0\noom 0\noom_kill 0\n",
        "memory.pressure": (
            "some avg10=0.10 avg60=0.20 avg300=0.30 total=4\n"
            "full avg10=0.01 avg60=0.02 avg300=0.03 total=1\n"
        ),
        "pids.current": "17\n",
        "pids.max": "max\n",
    }
    result = mut.parse_cgroup_snapshot(files)
    assert result["cgroup_version"] == "v2"
    assert result["memory_headroom_bytes"] == 180 * mut.GIB
    assert result["memory_high_bytes"] is None
    assert result["memory_peak_bytes"] is None
    assert result["missing_fields"] == ["memory.peak"]
    assert result["field_state"]["memory.peak"] == {
        "present": False,
        "raw": None,
    }
    assert result["events_high"] == 1
    assert result["pressure_full_avg10"] == pytest.approx(0.01)
    assert result["pids_max"] is None


def test_parse_cgroup_v1_never_invents_v2_oom_counters() -> None:
    files = {
        "memory.limit_in_bytes": "1000",
        "memory.soft_limit_in_bytes": "900",
        "memory.usage_in_bytes": "250",
        "memory.max_usage_in_bytes": "800",
        "memory.stat": "rss 100\ncache 50\ninactive_file 25\n",
        "memory.failcnt": "3",
        "memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n",
    }
    result = mut.parse_cgroup_snapshot(files, version="v1")
    assert result["memory_headroom_bytes"] == 750
    assert result["anon_bytes"] == 100
    assert result["file_bytes"] == 50
    assert result["events_oom"] is None
    assert "memory.pressure" in result["missing_fields"]
    assert result["memory_failcnt"] == 3


def test_static_440_gate_is_superseded_without_another_guessed_threshold() -> None:
    result = mut.supersede_static_440g_gate(
        authority=mut.EMPIRICAL_ADMISSION_AUTHORITY,
        old_required_free_gib=440,
    )
    assert result["state"] == "SUPERSEDED_UNMEASURED_STATIC_GATE"
    assert result["replacement_static_gate_bytes"] is None
    assert result["old_gate_enforced"] is False
    with pytest.raises(mut.MutFastAccurateV2Error, match="exact unmeasured"):
        mut.supersede_static_440g_gate(authority=True, old_required_free_gib=128)


def test_empirical_memory_formula_and_health_fail_closed() -> None:
    result = mut.derive_empirical_memory_admission(
        cgroup_memory_peak_bytes=20 * mut.GIB,
        process_peak_rss_bytes=18 * mut.GIB,
        checkpoint_peak_bytes=19 * mut.GIB,
    )
    assert result["status"] == "PASS"
    assert result["peak_gib"] == 20
    assert result["full_memory_max_gib"] == 76
    assert result["full_memory_high_gib"] == 57
    assert result["parent_required_headroom_gib"] == 92
    assert result["static_440_gib_gate_used"] is False

    blocked = mut.derive_empirical_memory_admission(
        cgroup_memory_peak_bytes=41 * mut.GIB,
        process_peak_rss_bytes=10 * mut.GIB,
        checkpoint_peak_bytes=10 * mut.GIB,
        memory_event_deltas={"oom_kill": 1},
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["full_memory_max_gib"] is None
    assert "UNEXPECTED_MUT_MEMORY_PEAK_GT_40_GIB" in blocked["blockers"]
    assert "CGROUP_MEMORY_LIMIT_EVENT_INCREASED" in blocked["blockers"]


def test_real_instrumentation_equivalence_schema_normalizes_to_semantic_pass() -> None:
    result = mut.validate_500_step_semantic_equivalence(
        _real_equivalence_receipt()
    )
    assert result["semantic_equivalence_pass"] is True
    assert result["equivalence_steps"] == 500
    assert result["checkpoint_resume_exercised"] is True

    failed = _real_equivalence_receipt()
    failed["checkpoint_resume_exercised"] = False
    with pytest.raises(mut.MutFastAccurateV2Error, match="checkpoint_resume"):
        mut.validate_500_step_semantic_equivalence(failed)


def test_historical_trace_on_50k_is_conditionally_adoptable_but_not_relabelled() -> None:
    result = mut.verify_historical_50k_artifact(
        _historical_manifest(),
        equivalence_receipt=_real_equivalence_receipt(),
        adoption_without_full_50k_parity_rerun_authorized=(
            mut.ADOPTION_WITHOUT_FULL_PARITY_AUTHORITY
        ),
        no_active_writer=True,
    )
    assert result["status"] == "PASS"
    assert result["trace_enabled"] is True
    assert result["trace_parity_passed"] is False
    assert result["traceoff_reference_rerun"] is False
    assert result["adoption_without_full_50k_parity_rerun_authorized"] is True
    assert result["target_parent_mismatch_count"] == 14
    assert result["cross_parent_convergence_count"] == 1
    assert result["target_parent_mismatch_nonfatal"] is True


def test_historical_adoption_rejects_missing_equivalence_and_lineage_errors() -> None:
    with pytest.raises(mut.MutFastAccurateV2Error, match="500-step"):
        mut.verify_historical_50k_artifact(
            _historical_manifest(),
            equivalence_receipt=None,
            adoption_without_full_50k_parity_rerun_authorized=True,
            no_active_writer=True,
        )
    changed = _historical_manifest()
    changed["lineage_recovery_audit"][
        "recorded_action_replay_mismatch_count"
    ] = 1
    with pytest.raises(mut.MutFastAccurateV2Error, match="lineage"):
        mut.verify_historical_50k_artifact(
            changed,
            equivalence_receipt=_real_equivalence_receipt(),
            adoption_without_full_50k_parity_rerun_authorized=True,
            no_active_writer=True,
        )


def test_transitive_candidate_pair_dbscan_binding_preserves_two_tiers() -> None:
    generation, pair, dbscan = _binding_evidence()
    result = mut.verify_candidate_universe_binding(generation, pair, dbscan)
    assert result["generation_payload_universe_sha256"] == SHA_A
    assert result["generation_payload_candidate_count"] == 100_235
    assert result["strict_flip_filtered_candidate_universe_sha256"] == SHA_C
    assert result["strict_flip_filtered_candidate_count"] == 50_620
    assert result["pair_store_recourse_vectors_sha256"] == SHA_D
    assert result["dbscan_native_candidate_universe_field_present"] is False
    assert result["claims_legacy_native_three_way_universe_fields"] is False


@pytest.mark.parametrize("mutation", ["payload", "manifest", "vectors", "universe"])
def test_transitive_binding_rejects_every_broken_edge(mutation: str) -> None:
    generation, pair, dbscan = _binding_evidence()
    if mutation == "payload":
        pair["scientific_identity"]["counterfactuals_sha256"] = SHA_E
    elif mutation == "manifest":
        pair["scientific_identity"]["generation_manifest_sha256"] = SHA_E
    elif mutation == "vectors":
        dbscan["scientific_identity"]["vectors_sha256"] = SHA_E
    else:
        dbscan["scientific_identity"]["source_candidate_universe_sha256"] = SHA_E
    with pytest.raises(mut.MutFastAccurateV2Error):
        mut.verify_candidate_universe_binding(generation, pair, dbscan)


def test_common_adoption_receipt_is_read_only_hash_closed_and_writer_free() -> None:
    binding = mut.verify_candidate_universe_binding(*_binding_evidence())
    receipt = {
        "status": "PASS",
        "pair_store_adopted_read_only": True,
        "dbscan_adopted_read_only": True,
        "common_recourse_adopted_read_only": True,
        "common_recourse_complete": True,
        "no_active_writer": True,
        "writable_fd_count": 0,
        "pair_store_rerun": False,
        "dbscan_rerun": False,
        "common_recourse_rerun": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "generation_payload_sha256": SHA_A,
        "candidate_universe_sha256": SHA_C,
        "pair_store_recourse_vectors_sha256": SHA_D,
        "dbscan_source_recourse_vectors_sha256": SHA_D,
        "selected_common_recourse_count": 100,
    }
    assert mut.validate_common_adoption_receipt(
        receipt, expected_binding=binding
    )["status"] == "PASS"
    receipt["writable_fd_count"] = 1
    with pytest.raises(mut.MutFastAccurateV2Error, match="writer"):
        mut.validate_common_adoption_receipt(receipt, expected_binding=binding)


def test_two_consecutive_train_only_windows_trigger_convergence() -> None:
    result = mut.evaluate_train_side_convergence(
        [_convergence_window(20_000), _convergence_window(22_500)]
    )
    assert result["status"] == "CONVERGED_EARLY_STOP"
    assert result["early_stop_used"] is True
    assert result["m_effective"] == 22_500
    assert result["stop_reason"] == "TRAIN_SIDE_CONVERGENCE"

    one_failure = _convergence_window(22_500)
    one_failure["top100_candidate_jaccard"] = 0.98
    continued = mut.evaluate_train_side_convergence(
        [_convergence_window(20_000), one_failure]
    )
    assert continued["status"] == "CONTINUE"
    assert continued["m_effective"] is None


def test_convergence_rejects_test_or_calibration_evidence() -> None:
    test_window = _convergence_window(20_000)
    test_window["test_loaded"] = True
    with pytest.raises(mut.MutFastAccurateV2Error, match="test_loaded"):
        mut.evaluate_train_side_convergence(
            [test_window, _convergence_window(22_500)]
        )


def test_capacity_report_distinguishes_inactive_capacity_and_frozen_eviction() -> None:
    inactive = mut.build_capacity_report(
        candidate_capacity=100_000,
        max_resident_candidate_count=50_620,
        capacity_eviction_count=0,
        candidate_count_at_stop=50_620,
    )
    assert inactive["capacity_reached"] is False
    assert inactive["capacity_constraint_inactive"] is True

    active = mut.build_capacity_report(
        candidate_capacity=100_000,
        max_resident_candidate_count=100_000,
        capacity_eviction_count=12,
        candidate_count_at_stop=100_012,
        eviction_policy="frequency_based_eviction",
    )
    assert active["capacity_reached"] is True
    assert active["capacity_constraint_inactive"] is False
    with pytest.raises(mut.MutFastAccurateV2Error, match="before capacity"):
        mut.build_capacity_report(
            candidate_capacity=100_000,
            max_resident_candidate_count=99_999,
            capacity_eviction_count=1,
            candidate_count_at_stop=100_000,
            eviction_policy="frequency_based_eviction",
        )


def test_cli_cgroup_snapshot_is_read_only_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text("1000\n", encoding="utf-8")
    (root / "memory.current").write_text("250\n", encoding="utf-8")
    assert mut.main(["cgroup-snapshot", "--root", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["memory_headroom_bytes"] == 750
    assert "memory.peak" in result["missing_fields"]
