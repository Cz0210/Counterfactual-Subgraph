from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pytest

import src.eval.tastemolnet_neurosed_fixed_budget as worker_gate
from src.eval.tastemolnet_neurosed_fixed_budget import (
    GEDLIB_WORKER_SELECTION_SCHEMA,
    NeuroSEDFixedBudgetError,
    blocked_gedlib_worker_resource_evidence,
    build_gedlib_worker_selection_manifest,
    select_fixed_pair_budget,
    select_gedlib_worker_count,
    summarize_real_gedlib_observations,
    validate_gedlib_worker_selection_manifest,
)


SHA = "a" * 64
COMMIT = "b" * 40
REPO_ROOT = Path(__file__).resolve().parents[2]


def _observations(
    prefix: str,
    count: int,
    *,
    timeouts: int = 0,
    failures: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        if index < timeouts:
            status = "TIMEOUT"
        elif index < timeouts + failures:
            status = "GEDLIB_ERROR"
        else:
            status = "SUCCESS"
        success = status == "SUCCESS"
        rows.append(
            {
                "pair_id": f"{prefix}-{index}",
                "status": status,
                "latency_seconds": 1.0,
                "lower_bound": 1.0 if success else None,
                "upper_bound": 2.0 if success else None,
                "exact_bound": False if success else None,
                "query_num_nodes": 5 + index % 3,
                "target_num_nodes": 8 + index % 5,
                "query_num_edges": 4 + index % 3,
                "target_num_edges": 7 + index % 5,
            }
        )
    return rows


def _report(
    prefix: str,
    *,
    workers: int,
    count: int = 100,
    wall_seconds: float = 100.0,
    timeouts: int = 0,
    failures: int = 0,
) -> dict:
    resource_evidence = blocked_gedlib_worker_resource_evidence(
        benchmark_process_identity={"pid": 123, "pid_start_ticks": 456},
        pre_sample={"unix_time_ns": 1, "iowait_ticks": 10},
        post_sample={"unix_time_ns": 2, "iowait_ticks": 11},
    )
    return summarize_real_gedlib_observations(
        _observations(prefix, count, timeouts=timeouts, failures=failures),
        benchmark_budget=count,
        worker_count=workers,
        wall_seconds=wall_seconds,
        pyged_module_sha256=SHA,
        gedlib_commit=COMMIT,
        gedlib_config_sha256=SHA,
        feature_schema_sha256=SHA,
        resource_metrics={
            "worker_resource_evidence": resource_evidence,
        },
    )


def _four_trials() -> dict[int, dict]:
    return {
        1: _report("worker-1", workers=1, wall_seconds=100.0),
        2: _report("worker-2", workers=2, wall_seconds=50.0),
        4: _report("worker-4", workers=4, wall_seconds=25.0),
        8: _report("worker-8", workers=8, wall_seconds=12.5),
    }


def _attach_reviewed_resource_evidence(
    report: dict,
    *,
    producer_sha256: str,
    bace_drop: float = 0.0,
    aids_drop: float = 0.0,
) -> None:
    evidence = report["resource_metrics"]["worker_resource_evidence"]
    evidence.update(
        {
            "status": "PASS",
            "producer_implemented": True,
            "producer_source_sha256": producer_sha256,
            "during_samples": [{"unix_time_ns": 2, "iowait_ticks": 10}],
            "protected_processes": {
                "bace_legacy": {"process_identity": {"pid": 11}},
                "aids_exact": {"process_identity": {"pid": 12}},
            },
            "missing_required_evidence": [],
            "host_load_gate_pass": True,
            "iowait_gate_pass": True,
            "bace_legacy_throughput_drop_percent": bace_drop,
            "aids_exact_throughput_drop_percent": aids_drop,
        }
    )
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_manifest_requires_all_physical_core_eligible_candidates() -> None:
    reports = _four_trials()
    reports.pop(4)
    with pytest.raises(NeuroSEDFixedBudgetError, match="every physical-core-eligible"):
        build_gedlib_worker_selection_manifest(reports, physical_core_count=8)

    two_core_manifest = build_gedlib_worker_selection_manifest(
        {1: reports[1], 2: reports[2]}, physical_core_count=2
    )
    assert two_core_manifest["required_worker_candidates"] == [1, 2]


def test_manifest_blocks_without_reviewed_resource_producer_and_is_replayable() -> None:
    manifest = build_gedlib_worker_selection_manifest(
        _four_trials(), physical_core_count=16
    )
    assert manifest["schema_version"] == GEDLIB_WORKER_SELECTION_SCHEMA
    assert manifest["status"] == "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED"
    assert manifest["selected_gedlib_workers"] is None
    assert manifest["manual_worker_override_used"] is False
    assert manifest["candidate_evidence"]["8"]["eligible"] is False
    assert manifest["candidate_evidence"]["8"]["exclusion_reasons"] == [
        "MACHINE_RESOURCE_EVIDENCE_UNAVAILABLE",
        "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED",
    ]
    assert validate_gedlib_worker_selection_manifest(manifest) == manifest
    with pytest.raises(NeuroSEDFixedBudgetError, match="no worker count"):
        select_gedlib_worker_count(_four_trials(), physical_core_count=16)


def test_timeout_failure_and_unhealthy_host_trials_are_excluded() -> None:
    reports = _four_trials()
    reports[2] = _report(
        "timeout-2", workers=2, wall_seconds=10.0, timeouts=1
    )
    reports[4] = _report(
        "failure-4", workers=4, wall_seconds=5.0, failures=1
    )
    reports[8] = _report("load-8", workers=8, wall_seconds=2.0)
    manifest = build_gedlib_worker_selection_manifest(
        reports, physical_core_count=8
    )
    assert manifest["selected_gedlib_workers"] is None
    assert manifest["candidate_evidence"]["2"]["exclusion_reasons"] == [
        "TIMEOUT_OBSERVED",
        "MACHINE_RESOURCE_EVIDENCE_UNAVAILABLE",
        "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED",
    ]
    assert manifest["candidate_evidence"]["4"]["exclusion_reasons"] == [
        "GEDLIB_ERROR_OBSERVED",
        "MACHINE_RESOURCE_EVIDENCE_UNAVAILABLE",
        "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED",
    ]
    assert manifest["candidate_evidence"]["8"]["exclusion_reasons"] == [
        "MACHINE_RESOURCE_EVIDENCE_UNAVAILABLE",
        "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED",
    ]


def test_worker_trials_must_be_pair_disjoint_and_use_one_backend() -> None:
    reports = _four_trials()
    reports[2]["pair_ids"][0] = reports[1]["pair_ids"][0]
    reports[2]["pair_ids_sha256"] = hashlib.sha256(
        json.dumps(
            reports[2]["pair_ids"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(NeuroSEDFixedBudgetError, match="reused"):
        build_gedlib_worker_selection_manifest(reports, physical_core_count=8)

    reports = _four_trials()
    reports[4]["gedlib_commit"] = "c" * 40
    with pytest.raises(NeuroSEDFixedBudgetError, match="backend/config changed"):
        build_gedlib_worker_selection_manifest(reports, physical_core_count=8)


def test_manifest_tamper_and_unreproducible_throughput_fail_closed() -> None:
    manifest = build_gedlib_worker_selection_manifest(
        _four_trials(), physical_core_count=8
    )
    manifest["selected_gedlib_workers"] = 8
    with pytest.raises(NeuroSEDFixedBudgetError, match="manifest changed"):
        validate_gedlib_worker_selection_manifest(manifest)

    reports = _four_trials()
    reports[4]["pairs_per_hour"] += 1.0
    with pytest.raises(NeuroSEDFixedBudgetError, match="cannot be reproduced"):
        build_gedlib_worker_selection_manifest(reports, physical_core_count=8)


def test_self_declared_machine_resource_pass_is_rejected() -> None:
    reports = _four_trials()
    evidence = reports[1]["resource_metrics"]["worker_resource_evidence"]
    evidence.update(
        {
            "status": "PASS",
            "producer_implemented": True,
            "producer_source_sha256": "c" * 64,
            "host_load_gate_pass": True,
            "iowait_gate_pass": True,
            "bace_legacy_throughput_drop_percent": 0.0,
            "aids_exact_throughput_drop_percent": 0.0,
        }
    )
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(NeuroSEDFixedBudgetError, match="no reviewed producer"):
        build_gedlib_worker_selection_manifest(reports, physical_core_count=8)


def test_one_missing_required_resource_authority_blocks_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_sha = "d" * 64
    monkeypatch.setattr(
        worker_gate,
        "REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256",
        producer_sha,
    )
    monkeypatch.setattr(worker_gate, "WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED", True)
    monkeypatch.setattr(
        worker_gate, "REVIEWED_WORKER_TRIAL_COHORT_BUILDER_SHA256", "e" * 64
    )
    reports = _four_trials()
    for workers in (1, 2, 4):
        _attach_reviewed_resource_evidence(
            reports[workers], producer_sha256=producer_sha
        )
    manifest = build_gedlib_worker_selection_manifest(
        reports, physical_core_count=8
    )
    assert manifest["status"] == "BLOCKED_GEDLIB_RESOURCE_EVIDENCE"
    assert manifest["all_required_resource_evidence_authenticated"] is False
    assert manifest["selected_gedlib_workers"] is None

    _attach_reviewed_resource_evidence(
        reports[8], producer_sha256=producer_sha, bace_drop=10.01
    )
    selected = build_gedlib_worker_selection_manifest(
        reports, physical_core_count=8
    )
    assert selected["status"] == "PASS"
    assert selected["all_required_resource_evidence_authenticated"] is True
    assert selected["selected_gedlib_workers"] == 4


def test_budget_planner_rejects_blocked_manifest_not_manual_integer() -> None:
    manifest = build_gedlib_worker_selection_manifest(
        _four_trials(), physical_core_count=8
    )
    benchmark = _report(
        "planning-1000", workers=4, count=1000, wall_seconds=1000.0
    )
    with pytest.raises(NeuroSEDFixedBudgetError, match="selection is not PASS"):
        select_fixed_pair_budget(
            benchmark,
            worker_selection_manifest=manifest,
            disk_reservation_pass=True,
        )
    assert "selected_workers" not in inspect.signature(select_fixed_pair_budget).parameters


def test_cli_and_paired_wrapper_expose_manifest_inputs_only() -> None:
    cli = (
        REPO_ROOT / "scripts/autodl/select_tastemolnet_neurosed_pair_budget.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        REPO_ROOT / "scripts/slurm/select_tastemolnet_neurosed_pair_budget.sh"
    ).read_text(encoding="utf-8")
    benchmark_cli = (
        REPO_ROOT / "scripts/autodl/benchmark_tastemolnet_neurosed_gedlib.py"
    ).read_text(encoding="utf-8")
    benchmark_wrapper = (
        REPO_ROOT / "scripts/slurm/benchmark_tastemolnet_neurosed_gedlib.sh"
    ).read_text(encoding="utf-8")
    assert "--worker-benchmark" in cli
    assert "--selected-workers" not in cli
    assert "--cpu-contention-gate-pass" not in cli
    assert "gedlib_worker_selection.json" in cli
    assert "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED" in wrapper
    assert "WORKER_RESOURCE_EVIDENCE_PRODUCER_NOT_IMPLEMENTED" in wrapper
    assert "--worker-benchmark" not in wrapper
    assert "--selected-workers" not in wrapper
    for manual_input in (
        "--bace-legacy-throughput-drop-percent",
        "--aids-exact-throughput-drop-percent",
        "--host-load-gate-pass",
        "--iowait-gate-pass",
    ):
        assert manual_input not in benchmark_cli
        assert manual_input not in benchmark_wrapper
    # The historical F2 worker-scaling route above remains fail-closed.  The
    # active main-table override is a fixed-worker, bounded non-MIP replay and
    # therefore must expose its backend/deadline and reap only owned children.
    assert "NON_MIP_METHOD_CONFIGS" in benchmark_cli
    assert "GED_LABEL_BACKEND_VARIANT" in benchmark_cli
    assert "--hard-wall-seconds" in benchmark_cli
    assert "process.terminate()" in benchmark_cli
    assert "process.kill()" in benchmark_cli
    assert "--method branch" in benchmark_wrapper

    blocker = json.loads(
        (
            REPO_ROOT
            / "configs/autodl/tastemolnet_neurosed_worker_trial_blockers_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert blocker["safe_to_select_workers"] is False
    assert blocker["marker"] is None
    assert (
        blocker["worker_trial_cohort_builder"]["status"]
        == "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED"
    )
