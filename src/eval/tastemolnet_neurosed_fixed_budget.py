"""Real-GEDLIB benchmark contracts and fixed NeuroSED pair-budget planning.

The helpers in this module never estimate a GED label and never turn a timeout
or backend error into training supervision.  They summarize observations made
by an authenticated ``pyged/GEDLIB`` runner and select only one of the approved
5k/10k/20k train budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.data.tastemolnet_neurosed_fixed_budget import (
    ALLOWED_TRAIN_PAIR_BUDGETS,
    BENCHMARK_BUDGETS,
    reserve_pair_count,
)


BENCHMARK_SCHEMA = "tastemolnet_neurosed_gedlib_benchmark_v1"
BENCHMARK_SUMMARY_SCHEMA = "tastemolnet_neurosed_gedlib_benchmark_summary_v1"
PAIR_BUDGET_PLAN_SCHEMA = "tastemolnet_neurosed_pair_budget_plan_v1"
GEDLIB_WORKER_SELECTION_SCHEMA = (
    "tastemolnet_neurosed_gedlib_worker_selection_v1"
)
GEDLIB_WORKER_RESOURCE_EVIDENCE_SCHEMA = (
    "tastemolnet_neurosed_gedlib_worker_resource_evidence_v1"
)
# No reviewed producer currently samples protected-job progress before/during
# each GEDLIB trial.  A self-declared PASS document must remain unusable until
# this pin is replaced by the reviewed producer source hash.
REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256: str | None = None
WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED = False
REVIEWED_WORKER_TRIAL_COHORT_BUILDER_SHA256: str | None = None
PAIR_LABELS_MANIFEST_SCHEMA = "tastemolnet_neurosed_pair_labels_manifest_v1"
OBSERVATION_STATUSES = frozenset({"SUCCESS", "TIMEOUT", "GEDLIB_ERROR"})
LEGAL_GEDLIB_WORKER_COUNTS = (1, 2, 4, 8)
OFFICIAL_SED_EDIT_COSTS = {
    "node_insertion": 0,
    "node_deletion": 1,
    "node_relabel_mismatch": 1,
    "edge_insertion": 0,
    "edge_deletion": 1,
    "edge_relabel": 0,
}
OFFICIAL_GED_DIRECTION = "query_to_target"


class NeuroSEDFixedBudgetError(RuntimeError):
    """A benchmark or resource plan cannot satisfy the fixed protocol."""


def _stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise NeuroSEDFixedBudgetError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NeuroSEDFixedBudgetError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise NeuroSEDFixedBudgetError(f"{label} must be finite and non-negative")
    return parsed


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise NeuroSEDFixedBudgetError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NeuroSEDFixedBudgetError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise NeuroSEDFixedBudgetError(f"{label} must be finite")
    return parsed


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise NeuroSEDFixedBudgetError("latency percentile requires observations")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denominator = sum((x - x_mean) ** 2 for x in xs)
    y_denominator = sum((y - y_mean) ** 2 for y in ys)
    if x_denominator <= 0 or y_denominator <= 0:
        return None
    return numerator / math.sqrt(x_denominator * y_denominator)


def summarize_real_gedlib_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    benchmark_budget: int,
    worker_count: int,
    wall_seconds: float,
    pyged_module_sha256: str,
    gedlib_commit: str,
    gedlib_config_sha256: str,
    feature_schema_sha256: str,
    pair_cohort_sha256: str | None = None,
    resource_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize raw observations from the authenticated real backend.

    Callers must provide one row for every attempted pair.  Successful rows
    carry both interval bounds; timeout/error rows must carry neither.
    """

    if int(benchmark_budget) not in BENCHMARK_BUDGETS and int(benchmark_budget) < 100:
        raise NeuroSEDFixedBudgetError("benchmark budget must be an approved tier or >=100")
    if len(observations) != int(benchmark_budget):
        raise NeuroSEDFixedBudgetError("benchmark observation count differs from budget")
    if int(worker_count) not in (1, 2, 4, 8):
        raise NeuroSEDFixedBudgetError("GEDLIB worker count must be 1, 2, 4, or 8")
    elapsed = _finite_nonnegative(wall_seconds, label="wall_seconds")
    if elapsed <= 0:
        raise NeuroSEDFixedBudgetError("wall_seconds must be positive")
    pair_ids: list[str] = []
    latencies: list[float] = []
    successful_latencies: list[float] = []
    node_complexities: list[float] = []
    edge_complexities: list[float] = []
    successful = 0
    exact_bounds = 0
    timeouts = 0
    failures = 0
    for index, observation in enumerate(observations):
        pair_id = str(observation.get("pair_id") or "")
        status = str(observation.get("status") or "")
        if not pair_id or status not in OBSERVATION_STATUSES:
            raise NeuroSEDFixedBudgetError(f"benchmark row {index} is malformed")
        pair_ids.append(pair_id)
        latency = _finite_nonnegative(
            observation.get("latency_seconds"), label=f"row {index} latency"
        )
        latencies.append(latency)
        if status == "SUCCESS":
            lower = _finite_nonnegative(
                observation.get("lower_bound"), label=f"row {index} lower_bound"
            )
            upper = _finite_nonnegative(
                observation.get("upper_bound"), label=f"row {index} upper_bound"
            )
            if lower > upper:
                raise NeuroSEDFixedBudgetError("GEDLIB returned lower_bound > upper_bound")
            exact_bound = observation.get("exact_bound")
            if type(exact_bound) is not bool or exact_bound is not (lower == upper):
                raise NeuroSEDFixedBudgetError("GEDLIB exact-bound flag changed")
            exact_bounds += int(exact_bound)
            successful += 1
            successful_latencies.append(latency)
            node_complexities.append(
                _finite_nonnegative(
                    observation.get("query_num_nodes"), label="query_num_nodes"
                )
                + _finite_nonnegative(
                    observation.get("target_num_nodes"), label="target_num_nodes"
                )
            )
            edge_complexities.append(
                _finite_nonnegative(
                    observation.get("query_num_edges"), label="query_num_edges"
                )
                + _finite_nonnegative(
                    observation.get("target_num_edges"), label="target_num_edges"
                )
            )
        else:
            if (
                observation.get("lower_bound") is not None
                or observation.get("upper_bound") is not None
            ):
                raise NeuroSEDFixedBudgetError(
                    "timeout/error observations must not retain a GED label"
                )
            if observation.get("exact_bound") is not None:
                raise NeuroSEDFixedBudgetError(
                    "timeout/error observations must not retain an exact-bound flag"
                )
            if status == "TIMEOUT":
                timeouts += 1
            else:
                failures += 1
    if len(set(pair_ids)) != len(pair_ids):
        raise NeuroSEDFixedBudgetError("benchmark pairs must be unique")
    if not successful_latencies:
        raise NeuroSEDFixedBudgetError("benchmark contains no successful GEDLIB pair")
    cohort_hash = _stable_sha256(pair_ids)
    if pair_cohort_sha256 is not None and str(pair_cohort_sha256) != cohort_hash:
        raise NeuroSEDFixedBudgetError("pair cohort SHA256 changed")
    for label, digest in (
        ("pyged_module_sha256", pyged_module_sha256),
        ("gedlib_config_sha256", gedlib_config_sha256),
        ("feature_schema_sha256", feature_schema_sha256),
    ):
        if len(str(digest)) != 64 or any(c not in "0123456789abcdef" for c in str(digest)):
            raise NeuroSEDFixedBudgetError(f"{label} must be a lowercase SHA256")
    if len(str(gedlib_commit)) != 40 or any(
        c not in "0123456789abcdef" for c in str(gedlib_commit)
    ):
        raise NeuroSEDFixedBudgetError("GEDLIB commit must be a full lowercase Git SHA")
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "status": "PASS" if failures == 0 else "PASS_WITH_GEDLIB_ERRORS",
        "real_pyged_gedlib_labels": True,
        "approximate_or_neural_labels_used": False,
        "ged_method": "f2",
        "ged_method_switched_from_official": False,
        "edit_cost_contract": dict(OFFICIAL_SED_EDIT_COSTS),
        "ged_direction": OFFICIAL_GED_DIRECTION,
        "label_representation": "lower_upper_interval_bounds",
        "label_transform": "none",
        "label_dtype": "float64_from_pyged",
        "benchmark_budget": int(benchmark_budget),
        "worker_count": int(worker_count),
        "attempted_pair_count": len(observations),
        "successful_pair_count": successful,
        "exact_bound_pair_count": exact_bounds,
        "interval_bound_pair_count": successful - exact_bounds,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / len(observations),
        "failure_count": failures,
        "wall_seconds": elapsed,
        "seconds_per_pair": elapsed / len(observations),
        "pairs_per_hour": successful * 3600.0 / elapsed,
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "successful_latency_seconds": {
            "p50": _percentile(successful_latencies, 0.50),
            "p90": _percentile(successful_latencies, 0.90),
            "p95": _percentile(successful_latencies, 0.95),
            "p99": _percentile(successful_latencies, 0.99),
        },
        "latency_query_target_node_count_correlation": _pearson(
            node_complexities, successful_latencies
        ),
        "latency_query_target_edge_count_correlation": _pearson(
            edge_complexities, successful_latencies
        ),
        "pair_ids_sha256": cohort_hash,
        "pair_ids": pair_ids,
        "pyged_module_sha256": str(pyged_module_sha256),
        "gedlib_commit": str(gedlib_commit),
        "gedlib_config_sha256": str(gedlib_config_sha256),
        "feature_schema_sha256": str(feature_schema_sha256),
        "resource_metrics": dict(resource_metrics or {}),
    }


def validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    expected_budget: int,
) -> dict[str, Any]:
    """Validate one report before it can influence a budget decision."""

    payload = dict(report)
    latency = payload.get("latency_seconds")
    if (
        payload.get("schema_version") != BENCHMARK_SCHEMA
        or payload.get("real_pyged_gedlib_labels") is not True
        or payload.get("approximate_or_neural_labels_used") is not False
        or payload.get("ged_method") != "f2"
        or payload.get("ged_method_switched_from_official") is not False
        or payload.get("edit_cost_contract") != OFFICIAL_SED_EDIT_COSTS
        or payload.get("ged_direction") != OFFICIAL_GED_DIRECTION
        or payload.get("label_representation") != "lower_upper_interval_bounds"
        or payload.get("label_transform") != "none"
        or payload.get("label_dtype") != "float64_from_pyged"
        or payload.get("benchmark_budget") != int(expected_budget)
        or payload.get("attempted_pair_count") != int(expected_budget)
        or type(payload.get("worker_count")) is not int
        or payload.get("worker_count") not in (1, 2, 4, 8)
        or type(latency) is not dict
    ):
        raise NeuroSEDFixedBudgetError("GEDLIB benchmark authority changed")
    _finite_nonnegative(latency.get("p95"), label="p95 latency")
    timeout_rate = _finite_nonnegative(payload.get("timeout_rate"), label="timeout rate")
    if timeout_rate > 1:
        raise NeuroSEDFixedBudgetError("timeout rate exceeds one")
    successful = payload.get("successful_pair_count")
    exact_bounds = payload.get("exact_bound_pair_count")
    interval_bounds = payload.get("interval_bound_pair_count")
    timeouts = payload.get("timeout_count")
    failures = payload.get("failure_count")
    if (
        type(successful) is not int
        or type(exact_bounds) is not int
        or type(interval_bounds) is not int
        or type(timeouts) is not int
        or type(failures) is not int
        or min(successful, exact_bounds, interval_bounds, timeouts, failures) < 0
        or exact_bounds + interval_bounds != successful
        or successful + timeouts + failures != int(expected_budget)
        or not math.isclose(timeout_rate, timeouts / int(expected_budget), abs_tol=1e-15)
        or payload.get("status")
        != ("PASS" if failures == 0 else "PASS_WITH_GEDLIB_ERRORS")
    ):
        raise NeuroSEDFixedBudgetError("benchmark outcome accounting changed")
    for label in (
        "pyged_module_sha256",
        "gedlib_config_sha256",
        "feature_schema_sha256",
    ):
        digest = str(payload.get(label) or "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise NeuroSEDFixedBudgetError(f"benchmark {label} is invalid")
    commit = str(payload.get("gedlib_commit") or "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise NeuroSEDFixedBudgetError("benchmark GEDLIB commit is invalid")
    pair_hash = str(payload.get("pair_ids_sha256") or "")
    if len(pair_hash) != 64 or any(c not in "0123456789abcdef" for c in pair_hash):
        raise NeuroSEDFixedBudgetError("benchmark pair cohort hash is invalid")
    pair_ids = payload.get("pair_ids")
    if (
        type(pair_ids) is not list
        or len(pair_ids) != int(expected_budget)
        or len(set(pair_ids)) != len(pair_ids)
        or _stable_sha256(pair_ids) != pair_hash
    ):
        raise NeuroSEDFixedBudgetError("benchmark pair ID inventory changed")
    return payload


def combine_disjoint_benchmark_reports(
    reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the disjoint 100/500/1000 reports into one benchmark authority."""

    if set(reports) != set(BENCHMARK_BUDGETS):
        raise NeuroSEDFixedBudgetError("benchmark summary requires 100/500/1000 tiers")
    checked = {
        budget: validate_benchmark_report(reports[budget], expected_budget=budget)
        for budget in BENCHMARK_BUDGETS
    }
    seen_pair_ids: set[str] = set()
    for budget in BENCHMARK_BUDGETS:
        current = set(checked[budget]["pair_ids"])
        if current.intersection(seen_pair_ids):
            raise NeuroSEDFixedBudgetError("benchmark tiers reused one or more pairs")
        seen_pair_ids.update(current)
    backend_fields = (
        "pyged_module_sha256",
        "gedlib_commit",
        "gedlib_config_sha256",
        "feature_schema_sha256",
        "ged_method",
        "edit_cost_contract",
        "ged_direction",
        "label_representation",
        "label_transform",
        "label_dtype",
    )
    reference = checked[100]
    for budget in (500, 1000):
        if any(checked[budget].get(field) != reference.get(field) for field in backend_fields):
            raise NeuroSEDFixedBudgetError("benchmark backend/config changed between tiers")
    failures = sum(int(checked[budget]["failure_count"]) for budget in BENCHMARK_BUDGETS)
    return {
        "schema_version": BENCHMARK_SUMMARY_SCHEMA,
        "status": "PASS" if failures == 0 else "BLOCKED_GEDLIB_ERRORS",
        "real_pyged_gedlib_labels": True,
        "gedlib_error_count": failures,
        "benchmark_budgets": list(BENCHMARK_BUDGETS),
        "benchmark_pair_cohorts_disjoint": True,
        "reports": {str(budget): checked[budget] for budget in BENCHMARK_BUDGETS},
        "summary_sha256": _stable_sha256(
            {str(budget): checked[budget] for budget in BENCHMARK_BUDGETS}
        ),
    }


def validation_pair_budget(train_pair_budget: int) -> int:
    if int(train_pair_budget) not in ALLOWED_TRAIN_PAIR_BUDGETS:
        raise NeuroSEDFixedBudgetError("train pair budget is not 5k/10k/20k")
    return min(4000, max(1000, math.floor(int(train_pair_budget) * 0.20)))


def _legal_gedlib_worker_candidates(physical_core_count: Any) -> tuple[int, ...]:
    if type(physical_core_count) is not int or physical_core_count < 1:
        raise NeuroSEDFixedBudgetError("physical core count must be a positive integer")
    return tuple(
        workers
        for workers in LEGAL_GEDLIB_WORKER_COUNTS
        if workers <= physical_core_count
    )


def blocked_gedlib_worker_resource_evidence(
    *,
    benchmark_process_identity: Mapping[str, Any],
    pre_sample: Mapping[str, Any],
    post_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only currently valid resource evidence: an explicit block."""

    payload: dict[str, Any] = {
        "schema_version": GEDLIB_WORKER_RESOURCE_EVIDENCE_SCHEMA,
        "status": "BLOCKED_RESOURCE_EVIDENCE_PRODUCER_MISSING",
        "marker": None,
        "machine_generated": True,
        "producer_implemented": False,
        "producer_source_sha256": None,
        "benchmark_process_identity": dict(benchmark_process_identity),
        "pre_sample": dict(pre_sample),
        "during_samples": [],
        "post_sample": dict(post_sample),
        "protected_processes": {
            "bace_legacy": None,
            "aids_exact": None,
        },
        "missing_required_evidence": [
            "authenticated_bace_legacy_process_identity",
            "authenticated_aids_exact_process_identity",
            "bace_pre_and_during_progress_counters",
            "aids_pre_and_during_progress_counters",
            "periodic_load_and_iowait_samples",
            "reviewed_resource_evidence_producer_source",
        ],
        "host_load_gate_pass": None,
        "iowait_gate_pass": None,
        "bace_legacy_throughput_drop_percent": None,
        "aids_exact_throughput_drop_percent": None,
    }
    payload["evidence_sha256"] = _stable_sha256(payload)
    return payload


def _validate_worker_resource_evidence(
    resources: Any,
) -> dict[str, Any]:
    """Validate a resource-evidence binding or return an explicit blocker."""

    if type(resources) is not dict:
        return {
            "status": "MISSING",
            "evidence_sha256": None,
            "eligible": False,
        }
    evidence = resources.get("worker_resource_evidence")
    if type(evidence) is not dict:
        return {
            "status": "MISSING",
            "evidence_sha256": None,
            "eligible": False,
        }
    if evidence.get("schema_version") != GEDLIB_WORKER_RESOURCE_EVIDENCE_SCHEMA:
        raise NeuroSEDFixedBudgetError("worker resource evidence schema changed")
    recorded_sha256 = str(evidence.get("evidence_sha256") or "")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    if (
        len(recorded_sha256) != 64
        or any(character not in "0123456789abcdef" for character in recorded_sha256)
        or recorded_sha256 != _stable_sha256(unsigned)
    ):
        raise NeuroSEDFixedBudgetError("worker resource evidence hash changed")
    status = evidence.get("status")
    if status == "PASS":
        if REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256 is None:
            raise NeuroSEDFixedBudgetError(
                "worker resource evidence PASS has no reviewed producer"
            )
        required = (
            "pre_sample",
            "during_samples",
            "post_sample",
            "benchmark_process_identity",
            "protected_processes",
        )
        if any(evidence.get(field) is None for field in required):
            raise NeuroSEDFixedBudgetError(
                "worker resource evidence lacks machine samples or process identity"
            )
        if (
            evidence.get("producer_source_sha256")
            != REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256
        ):
            raise NeuroSEDFixedBudgetError(
                "worker resource evidence producer source changed"
            )
        return {
            "status": "PASS",
            "evidence_sha256": recorded_sha256,
            "eligible": True,
            "host_load_gate_pass": evidence.get("host_load_gate_pass"),
            "iowait_gate_pass": evidence.get("iowait_gate_pass"),
            "bace_legacy_throughput_drop_percent": evidence.get(
                "bace_legacy_throughput_drop_percent"
            ),
            "aids_exact_throughput_drop_percent": evidence.get(
                "aids_exact_throughput_drop_percent"
            ),
        }
    if (
        status != "BLOCKED_RESOURCE_EVIDENCE_PRODUCER_MISSING"
        or evidence.get("producer_implemented") is not False
        or evidence.get("producer_source_sha256") is not None
    ):
        raise NeuroSEDFixedBudgetError("worker resource evidence status changed")
    if (
        evidence.get("machine_generated") is not True
        or type(evidence.get("benchmark_process_identity")) is not dict
        or type(evidence.get("pre_sample")) is not dict
        or type(evidence.get("during_samples")) is not list
        or type(evidence.get("post_sample")) is not dict
        or evidence.get("protected_processes")
        != {"bace_legacy": None, "aids_exact": None}
        or type(evidence.get("missing_required_evidence")) is not list
        or not evidence.get("missing_required_evidence")
    ):
        raise NeuroSEDFixedBudgetError(
            "blocked worker resource evidence manifest changed"
        )
    return {
        "status": status,
        "evidence_sha256": recorded_sha256,
        "eligible": False,
    }


def build_gedlib_worker_selection_manifest(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    physical_core_count: int,
) -> dict[str, Any]:
    """Recompute a complete worker choice from real, disjoint GEDLIB trials.

    Every legal worker count available on the current physical host must have
    one real report with at least 100 unique pairs.  Unhealthy trials remain in
    the manifest as evidence but cannot participate in throughput ranking.
    """

    candidates = _legal_gedlib_worker_candidates(physical_core_count)
    if type(reports) is not dict or set(reports) != set(candidates):
        raise NeuroSEDFixedBudgetError(
            "worker selection requires every physical-core-eligible 1/2/4/8 trial"
        )
    checked_reports: dict[str, dict[str, Any]] = {}
    candidate_evidence: dict[str, dict[str, Any]] = {}
    seen_pair_ids: set[str] = set()
    backend_reference: dict[str, Any] | None = None
    eligible: list[tuple[float, int]] = []
    backend_fields = (
        "pyged_module_sha256",
        "gedlib_commit",
        "gedlib_config_sha256",
        "feature_schema_sha256",
        "ged_method",
        "edit_cost_contract",
        "ged_direction",
        "label_representation",
        "label_transform",
        "label_dtype",
    )
    for workers in candidates:
        raw = reports[workers]
        budget = raw.get("benchmark_budget")
        if type(budget) is not int or budget < 100:
            raise NeuroSEDFixedBudgetError(
                "worker trial must contain at least 100 fresh real pairs"
            )
        report = validate_benchmark_report(raw, expected_budget=budget)
        if report["worker_count"] != workers:
            raise NeuroSEDFixedBudgetError("worker trial count differs from candidate")
        cohort = set(report["pair_ids"])
        if cohort.intersection(seen_pair_ids):
            raise NeuroSEDFixedBudgetError("worker trials reused benchmark pairs")
        seen_pair_ids.update(cohort)

        backend = {field: report.get(field) for field in backend_fields}
        if backend_reference is None:
            backend_reference = backend
        elif backend != backend_reference:
            raise NeuroSEDFixedBudgetError(
                "worker trial backend/config changed between candidates"
            )

        wall_seconds = _finite_nonnegative(
            report.get("wall_seconds"), label="worker trial wall_seconds"
        )
        if wall_seconds <= 0:
            raise NeuroSEDFixedBudgetError("worker trial wall_seconds must be positive")
        throughput = _finite_nonnegative(
            report.get("pairs_per_hour"), label="worker trial pairs/hour"
        )
        recomputed_throughput = (
            int(report["successful_pair_count"]) * 3600.0 / wall_seconds
        )
        if not math.isclose(
            throughput, recomputed_throughput, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise NeuroSEDFixedBudgetError("worker trial throughput cannot be reproduced")

        resource_evidence = _validate_worker_resource_evidence(
            report.get("resource_metrics")
        )
        host_load_pass: bool | None = None
        iowait_pass: bool | None = None
        bace_drop: float | None = None
        aids_drop: float | None = None
        if resource_evidence["eligible"]:
            host_load_pass = resource_evidence.get("host_load_gate_pass")
            iowait_pass = resource_evidence.get("iowait_gate_pass")
            if type(host_load_pass) is not bool or type(iowait_pass) is not bool:
                raise NeuroSEDFixedBudgetError(
                    "worker resource load/iowait gates must be machine booleans"
                )
            bace_drop = _finite(
                resource_evidence.get("bace_legacy_throughput_drop_percent"),
                label="BACE legacy throughput drop percent",
            )
            aids_drop = _finite(
                resource_evidence.get("aids_exact_throughput_drop_percent"),
                label="AIDS exact throughput drop percent",
            )

        exclusion_reasons: list[str] = []
        if int(report["timeout_count"]) != 0:
            exclusion_reasons.append("TIMEOUT_OBSERVED")
        if int(report["failure_count"]) != 0:
            exclusion_reasons.append("GEDLIB_ERROR_OBSERVED")
        if not resource_evidence["eligible"]:
            exclusion_reasons.append("MACHINE_RESOURCE_EVIDENCE_UNAVAILABLE")
        else:
            if host_load_pass is not True:
                exclusion_reasons.append("HOST_LOAD_UNHEALTHY")
            if iowait_pass is not True:
                exclusion_reasons.append("IOWAIT_UNHEALTHY")
            if bace_drop is not None and bace_drop > 10.0:
                exclusion_reasons.append("BACE_LEGACY_DROP_GT_10PCT")
            if aids_drop is not None and aids_drop > 10.0:
                exclusion_reasons.append("AIDS_EXACT_DROP_GT_10PCT")
        if not WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED:
            exclusion_reasons.append("WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED")
        is_eligible = not exclusion_reasons
        if is_eligible:
            eligible.append((throughput, workers))
        checked_reports[str(workers)] = report
        candidate_evidence[str(workers)] = {
            "worker_count": workers,
            "benchmark_budget": budget,
            "fresh_unique_pair_count": len(cohort),
            "pair_ids_sha256": report["pair_ids_sha256"],
            "report_sha256": _stable_sha256(report),
            "pairs_per_hour": throughput,
            "timeout_count": int(report["timeout_count"]),
            "gedlib_error_count": int(report["failure_count"]),
            "worker_resource_evidence_status": resource_evidence["status"],
            "worker_resource_evidence_sha256": resource_evidence[
                "evidence_sha256"
            ],
            "host_load_gate_pass": host_load_pass,
            "iowait_gate_pass": iowait_pass,
            "bace_legacy_throughput_drop_percent": bace_drop,
            "aids_exact_throughput_drop_percent": aids_drop,
            "contention_drop_limit_percent": 10.0,
            "eligible": is_eligible,
            "exclusion_reasons": exclusion_reasons,
        }

    eligible.sort(key=lambda item: (-item[0], item[1]))
    resource_evidence_authority_complete = all(
        row["worker_resource_evidence_status"] == "PASS"
        for row in candidate_evidence.values()
    )
    selected_workers = (
        eligible[0][1]
        if eligible
        and resource_evidence_authority_complete
        and WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED
        else None
    )
    if not WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED:
        status = "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED"
    elif not resource_evidence_authority_complete:
        status = "BLOCKED_GEDLIB_RESOURCE_EVIDENCE"
    elif selected_workers is not None:
        status = "PASS"
    else:
        status = "BLOCKED_GEDLIB_WORKER_SELECTION"
    payload: dict[str, Any] = {
        "schema_version": GEDLIB_WORKER_SELECTION_SCHEMA,
        "status": status,
        "selection_policy": "highest_healthy_real_pairs_per_hour_then_lower_worker_count",
        "physical_core_count": physical_core_count,
        "physical_core_count_source": "runtime_physical_core_probe",
        "legal_worker_counts": list(LEGAL_GEDLIB_WORKER_COUNTS),
        "required_worker_candidates": list(candidates),
        "all_required_candidate_reports_present": True,
        "minimum_fresh_pairs_per_candidate": 100,
        "worker_trial_pair_cohorts_disjoint": True,
        "worker_trial_cohort_builder_implemented": (
            WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED
        ),
        "worker_trial_cohort_builder_sha256": (
            REVIEWED_WORKER_TRIAL_COHORT_BUILDER_SHA256
        ),
        "worker_trial_cohort_builder_status": (
            "IMPLEMENTED"
            if WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED
            else "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED"
        ),
        "timeout_or_failure_candidates_excluded": True,
        "contention_drop_limit_percent": 10.0,
        "manual_worker_override_used": False,
        "resource_evidence_producer_implemented": (
            REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256 is not None
        ),
        "all_required_resource_evidence_authenticated": (
            resource_evidence_authority_complete
        ),
        "resource_evidence_producer_sha256": (
            REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256
        ),
        "resource_evidence_blocker": (
            "reviewed protected-job resource/throughput evidence producer is missing"
            if REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256 is None
            else None
        ),
        "infrastructure_blockers": [
            value
            for value in (
                (
                    None
                    if WORKER_TRIAL_COHORT_BUILDER_IMPLEMENTED
                    else "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED"
                ),
                (
                    None
                    if REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256 is not None
                    else "WORKER_RESOURCE_EVIDENCE_PRODUCER_NOT_IMPLEMENTED"
                ),
            )
            if value is not None
        ],
        "backend_authority": backend_reference,
        "candidate_evidence": candidate_evidence,
        "reports": checked_reports,
        "selected_gedlib_workers": selected_workers,
    }
    payload["manifest_sha256"] = _stable_sha256(payload)
    return payload


def validate_gedlib_worker_selection_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild a worker manifest so a selected integer is never trusted."""

    payload = dict(manifest)
    if payload.get("schema_version") != GEDLIB_WORKER_SELECTION_SCHEMA:
        raise NeuroSEDFixedBudgetError("GEDLIB worker selection schema changed")
    reports = payload.get("reports")
    if type(reports) is not dict:
        raise NeuroSEDFixedBudgetError("GEDLIB worker selection reports are missing")
    parsed_reports: dict[int, Mapping[str, Any]] = {}
    for key, report in reports.items():
        if type(key) is not str or not key.isdigit() or type(report) is not dict:
            raise NeuroSEDFixedBudgetError("GEDLIB worker selection report map changed")
        parsed_reports[int(key)] = report
    expected = build_gedlib_worker_selection_manifest(
        parsed_reports,
        physical_core_count=payload.get("physical_core_count"),
    )
    if payload != expected:
        raise NeuroSEDFixedBudgetError("GEDLIB worker selection manifest changed")
    return payload


def select_gedlib_worker_count(
    reports: Mapping[int, Mapping[str, Any]],
    *,
    physical_core_count: int,
) -> int:
    """Return the machine-selected worker count from the strict manifest."""

    manifest = build_gedlib_worker_selection_manifest(
        reports, physical_core_count=physical_core_count
    )
    selected = manifest["selected_gedlib_workers"]
    if selected is None:
        raise NeuroSEDFixedBudgetError("no worker count passed contention gates")
    return int(selected)


def select_fixed_pair_budget(
    benchmark_1000: Mapping[str, Any],
    *,
    worker_selection_manifest: Mapping[str, Any],
    disk_reservation_pass: bool,
    maximum_label_hours: float = 24.0,
    timeout_rate_maximum: float = 0.05,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Choose a fixed budget from one revalidated machine worker manifest."""

    report = validate_benchmark_report(benchmark_1000, expected_budget=1000)
    worker_manifest = validate_gedlib_worker_selection_manifest(
        worker_selection_manifest
    )
    if worker_manifest["status"] != "PASS":
        raise NeuroSEDFixedBudgetError("GEDLIB worker selection is not PASS")
    selected_workers = int(worker_manifest["selected_gedlib_workers"])
    if selected_workers != report["worker_count"]:
        raise NeuroSEDFixedBudgetError(
            "1000-pair benchmark worker count differs from machine selection"
        )
    backend = worker_manifest["backend_authority"]
    if any(report.get(field) != value for field, value in backend.items()):
        raise NeuroSEDFixedBudgetError(
            "1000-pair benchmark backend differs from worker-selection trials"
        )
    hours_limit = _finite_nonnegative(maximum_label_hours, label="maximum label hours")
    timeout_limit = _finite_nonnegative(
        timeout_rate_maximum, label="timeout rate maximum"
    )
    factor = _finite_nonnegative(safety_factor, label="safety factor")
    if hours_limit <= 0 or factor < 1 or timeout_limit > 1:
        raise NeuroSEDFixedBudgetError("resource-gate constants are invalid")
    p95 = _finite_nonnegative(
        report["latency_seconds"]["p95"], label="1000-pair p95 latency"
    )
    if p95 <= 0:
        raise NeuroSEDFixedBudgetError("1000-pair p95 latency must be positive")
    projections: dict[str, dict[str, Any]] = {}
    selected: int | None = None
    for train_budget in ALLOWED_TRAIN_PAIR_BUDGETS:
        val_budget = validation_pair_budget(train_budget)
        total_pairs = train_budget + val_budget
        hours = total_pairs * p95 / int(selected_workers) * factor / 3600.0
        projections[str(train_budget)] = {
            "train_pair_budget": train_budget,
            "validation_pair_budget": val_budget,
            "total_pair_budget": total_pairs,
            "projected_label_hours": hours,
            "within_24_hour_gate": hours <= hours_limit,
        }
    all_common_gates = (
        report["timeout_rate"] <= timeout_limit
        and report["failure_count"] == 0
        and disk_reservation_pass is True
        and worker_manifest["status"] == "PASS"
    )
    if all_common_gates:
        for train_budget in reversed(ALLOWED_TRAIN_PAIR_BUDGETS):
            if projections[str(train_budget)]["within_24_hour_gate"]:
                selected = train_budget
                break
    status = "PASS" if selected is not None else "BLOCKED_GEDLIB_THROUGHPUT"
    payload = {
        "schema_version": PAIR_BUDGET_PLAN_SCHEMA,
        "status": status,
        "pair_budget_strategy": "fixed_budget_resource_control",
        "official_pair_semantics_changed_by_budget": False,
        "allowed_train_pair_budgets": list(ALLOWED_TRAIN_PAIR_BUDGETS),
        "maximum_label_hours": hours_limit,
        "timeout_rate_maximum": timeout_limit,
        "safety_factor": factor,
        "ged_p95_seconds_per_pair": p95,
        "selected_gedlib_workers": int(selected_workers),
        "worker_selection_manifest_sha256": worker_manifest["manifest_sha256"],
        "worker_selection_policy": worker_manifest["selection_policy"],
        "manual_worker_override_used": False,
        "observed_timeout_rate": report["timeout_rate"],
        "observed_gedlib_error_count": report["failure_count"],
        "disk_reservation_pass": disk_reservation_pass is True,
        "cpu_contention_gate_pass": True,
        "projections": projections,
        "selected_neurosed_train_pair_budget": selected,
        "selected_neurosed_validation_pair_budget": (
            validation_pair_budget(selected) if selected is not None else None
        ),
        "shortcut_fallback_used": False,
        "approximate_label_fallback_used": False,
    }
    payload["plan_sha256"] = _stable_sha256(payload)
    return payload


def directional_ged_cache_key(
    *,
    query_canonical_graph_sha256: str,
    target_canonical_graph_sha256: str,
    gedlib_config_sha256: str,
    feature_schema_sha256: str,
    direction: str = OFFICIAL_GED_DIRECTION,
) -> str:
    """Bind an ordered SED cache key; reverse pairs never share an entry."""

    policy = ged_cache_symmetry_policy(OFFICIAL_SED_EDIT_COSTS)
    if policy["symmetric"] is not False or policy["share_reverse_cache"] is not False:
        raise NeuroSEDFixedBudgetError("official SED unexpectedly became symmetric")
    return ged_cache_key(
        query_canonical_graph_sha256=query_canonical_graph_sha256,
        target_canonical_graph_sha256=target_canonical_graph_sha256,
        gedlib_config_sha256=gedlib_config_sha256,
        feature_schema_sha256=feature_schema_sha256,
        edit_cost_contract=OFFICIAL_SED_EDIT_COSTS,
        direction=direction,
    )


def ged_cache_symmetry_policy(edit_cost_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Determine reverse-cache safety from the complete scalar cost contract."""

    expected = set(OFFICIAL_SED_EDIT_COSTS)
    if set(edit_cost_contract) != expected:
        raise NeuroSEDFixedBudgetError("GED cache edit-cost fields changed")
    costs = {
        key: _finite_nonnegative(edit_cost_contract[key], label=f"{key} cost")
        for key in sorted(expected)
    }
    symmetric = (
        costs["node_insertion"] == costs["node_deletion"]
        and costs["edge_insertion"] == costs["edge_deletion"]
    )
    return {
        "schema_version": "tastemolnet_neurosed_ged_cache_policy_v1",
        "symmetric": symmetric,
        "share_reverse_cache": symmetric,
        "query_target_order_in_cache_key": not symmetric,
        "edit_cost_contract": costs,
        "policy_reason": (
            "insertion_and_deletion_costs_match"
            if symmetric
            else "insertion_and_deletion_costs_differ"
        ),
    }


def ged_cache_key(
    *,
    query_canonical_graph_sha256: str,
    target_canonical_graph_sha256: str,
    gedlib_config_sha256: str,
    feature_schema_sha256: str,
    edit_cost_contract: Mapping[str, Any],
    direction: str = OFFICIAL_GED_DIRECTION,
) -> str:
    """Build a symmetric or ordered key only after explicit cost inspection."""

    digests = {
        "query_canonical_graph_sha256": query_canonical_graph_sha256,
        "target_canonical_graph_sha256": target_canonical_graph_sha256,
        "gedlib_config_sha256": gedlib_config_sha256,
        "feature_schema_sha256": feature_schema_sha256,
    }
    for label, digest in digests.items():
        if len(str(digest)) != 64 or any(
            character not in "0123456789abcdef" for character in str(digest)
        ):
            raise NeuroSEDFixedBudgetError(f"{label} is not a lowercase SHA256")
    if direction != OFFICIAL_GED_DIRECTION:
        raise NeuroSEDFixedBudgetError("GED cache direction changed")
    policy = ged_cache_symmetry_policy(edit_cost_contract)
    if policy["symmetric"]:
        query_hash, target_hash = sorted(
            (str(query_canonical_graph_sha256), str(target_canonical_graph_sha256))
        )
        role_binding = {
            "unordered_graph_sha256": [query_hash, target_hash],
            "direction": "symmetric",
        }
    else:
        role_binding = {
            "query_canonical_graph_sha256": str(query_canonical_graph_sha256),
            "target_canonical_graph_sha256": str(target_canonical_graph_sha256),
            "direction": direction,
        }
    return _stable_sha256(
        {
            **role_binding,
            "gedlib_config_sha256": str(gedlib_config_sha256),
            "feature_schema_sha256": str(feature_schema_sha256),
            "edit_cost_contract": policy["edit_cost_contract"],
        }
    )


def select_successful_reserve_pairs(
    observations: Sequence[Mapping[str, Any]],
    *,
    requested_pair_count: int,
) -> dict[str, Any]:
    """Take the first successful rows in sampler order, never by GED value."""

    requested = int(requested_pair_count)
    expected_candidates = reserve_pair_count(requested)
    if len(observations) != expected_candidates:
        raise NeuroSEDFixedBudgetError(
            "reserve observation count must equal ceil(1.10 * requested budget)"
        )
    selected: list[str] = []
    seen: set[str] = set()
    attempted = 0
    timeouts = 0
    errors = 0
    for index, row in enumerate(observations):
        pair_id = str(row.get("pair_id") or "")
        status = str(row.get("status") or "")
        if not pair_id or pair_id in seen or status not in OBSERVATION_STATUSES:
            raise NeuroSEDFixedBudgetError(f"reserve row {index} is malformed")
        seen.add(pair_id)
        attempted += 1
        if status == "SUCCESS":
            lower = _finite_nonnegative(row.get("lower_bound"), label="reserve lower bound")
            upper = _finite_nonnegative(row.get("upper_bound"), label="reserve upper bound")
            if lower > upper:
                raise NeuroSEDFixedBudgetError("reserve lower bound exceeds upper bound")
            selected.append(pair_id)
        else:
            if row.get("lower_bound") is not None or row.get("upper_bound") is not None:
                raise NeuroSEDFixedBudgetError(
                    "timeout/error reserve row must not carry a GED label"
                )
            if status == "TIMEOUT":
                timeouts += 1
            else:
                errors += 1
        if len(selected) == requested:
            break
    status = "PASS" if len(selected) == requested else "BLOCKED_GEDLIB_LABEL_YIELD"
    return {
        "schema_version": "tastemolnet_neurosed_reserve_selection_v1",
        "status": status,
        "requested_pair_count": requested,
        "reserve_candidate_count": expected_candidates,
        "attempted_pair_count": attempted,
        "successful_pair_count": len(selected),
        "timeout_count": timeouts,
        "error_count": errors,
        "reserve_used": max(0, attempted - requested),
        "selected_pair_ids": selected,
        "selected_in_sampler_order": True,
        "ged_value_based_selection_used": False,
        "approximate_label_fallback_used": False,
    }


def _float32(value: float, *, label: str) -> float:
    try:
        result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise NeuroSEDFixedBudgetError(f"{label} cannot be stored as float32") from exc
    if not math.isfinite(result) or result < 0:
        raise NeuroSEDFixedBudgetError(f"{label} float32 value is invalid")
    return result


def official_ged_interval_label(
    observation: Mapping[str, Any],
    *,
    gedlib_commit: str,
    pyged_module_sha256: str,
    gedlib_config_sha256: str,
    feature_schema_sha256: str,
    pair_sampler_manifest_sha256: str,
    gedlib_build_manifest_sha256: str,
    ged_method_args: str,
) -> dict[str, Any]:
    """Convert one successful real-pyged result to the official training row."""

    if observation.get("status") != "SUCCESS":
        raise NeuroSEDFixedBudgetError("only SUCCESS observations may become labels")
    pair_id = str(observation.get("pair_id") or "")
    if not pair_id:
        raise NeuroSEDFixedBudgetError("GED label pair_id is empty")
    query_graph_id = str(observation.get("query_graph_id") or "")
    target_graph_id = str(observation.get("target_graph_id") or "")
    query_split = str(observation.get("query_split") or "")
    target_split = str(observation.get("target_split") or "")
    if (
        not query_graph_id
        or not target_graph_id
        or query_graph_id == target_graph_id
        or query_split not in ("train", "validation")
        or target_split != query_split
    ):
        raise NeuroSEDFixedBudgetError("GED label pair role/split contract changed")
    lower_raw = _finite_nonnegative(
        observation.get("lower_bound"), label="pyged lower bound"
    )
    upper_raw = _finite_nonnegative(
        observation.get("upper_bound"), label="pyged upper bound"
    )
    if lower_raw > upper_raw:
        raise NeuroSEDFixedBudgetError("pyged lower bound exceeds upper bound")
    exact = observation.get("exact_bound")
    if type(exact) is not bool or exact is not (lower_raw == upper_raw):
        raise NeuroSEDFixedBudgetError("pyged exact/bound flag changed")
    canonical_query = str(observation.get("query_canonical_graph_sha256") or "")
    canonical_target = str(observation.get("target_canonical_graph_sha256") or "")
    cache_key = directional_ged_cache_key(
        query_canonical_graph_sha256=canonical_query,
        target_canonical_graph_sha256=canonical_target,
        gedlib_config_sha256=gedlib_config_sha256,
        feature_schema_sha256=feature_schema_sha256,
    )
    for label, digest in (
        ("pyged_module_sha256", pyged_module_sha256),
        ("gedlib_config_sha256", gedlib_config_sha256),
        ("feature_schema_sha256", feature_schema_sha256),
        ("pair_sampler_manifest_sha256", pair_sampler_manifest_sha256),
        ("gedlib_build_manifest_sha256", gedlib_build_manifest_sha256),
    ):
        if len(str(digest)) != 64 or any(
            character not in "0123456789abcdef" for character in str(digest)
        ):
            raise NeuroSEDFixedBudgetError(f"{label} is invalid")
    if len(str(gedlib_commit)) != 40 or any(
        character not in "0123456789abcdef" for character in str(gedlib_commit)
    ):
        raise NeuroSEDFixedBudgetError("GEDLIB label commit is invalid")
    if ged_method_args != "--threads 1 --time-limit 1":
        raise NeuroSEDFixedBudgetError("official F2 GED method arguments changed")
    lower = _float32(lower_raw, label="lower bound")
    upper = _float32(upper_raw, label="upper bound")
    if lower > upper:
        raise NeuroSEDFixedBudgetError("float32 lower bound exceeds upper bound")
    return {
        "schema_version": "tastemolnet_neurosed_official_ged_label_v1",
        "pair_id": pair_id,
        "query_graph_id": query_graph_id,
        "target_graph_id": target_graph_id,
        "query_split": query_split,
        "target_split": target_split,
        "status": "SUCCESS",
        "lower_bound": lower,
        "upper_bound": upper,
        "exact_bound": exact,
        "stored_bounds_equal": lower == upper,
        "bounds_kind": "exact" if exact else "interval",
        "ged_method": "f2",
        "ged_method_args": ged_method_args,
        "gedlib_commit": str(gedlib_commit),
        "pyged_module_sha256": str(pyged_module_sha256),
        "gedlib_config_sha256": str(gedlib_config_sha256),
        "feature_schema_sha256": str(feature_schema_sha256),
        "pair_sampler_manifest_sha256": str(pair_sampler_manifest_sha256),
        "gedlib_build_manifest_sha256": str(gedlib_build_manifest_sha256),
        "query_canonical_graph_sha256": canonical_query,
        "target_canonical_graph_sha256": canonical_target,
        "direction": OFFICIAL_GED_DIRECTION,
        "cache_key": cache_key,
        "cache_symmetric": False,
        "reverse_cache_shared": False,
        "query_target_order_in_cache_key": True,
        "pyged_return_dtype": "float64",
        "label_dtype": "float32",
        "label_transform": "official_torch_float32_storage_cast_only",
        "target_representation": "ordered_query_target_lower_upper_interval",
        "bound_average_used": False,
        "single_bound_substitution_used": False,
        "approximate_or_neural_label_used": False,
    }


def build_official_pair_labels_manifest(
    labels: Sequence[Mapping[str, Any]],
    *,
    split: str,
    requested_pair_count: int,
    reserve_selection: Mapping[str, Any],
    compact_storage_format: str,
    compact_labels_sha256: str,
) -> dict[str, Any]:
    """Close selected real labels without publishing a large JSON row dump."""

    if split not in ("train", "validation"):
        raise NeuroSEDFixedBudgetError("label split must be train or validation")
    requested = int(requested_pair_count)
    if requested <= 0 or len(labels) != requested:
        raise NeuroSEDFixedBudgetError("successful label count differs from budget")
    if compact_storage_format not in ("parquet", "arrow_ipc", "numpy_npz"):
        raise NeuroSEDFixedBudgetError("full labels must use compact columnar/binary storage")
    if len(str(compact_labels_sha256)) != 64 or any(
        character not in "0123456789abcdef"
        for character in str(compact_labels_sha256)
    ):
        raise NeuroSEDFixedBudgetError("compact label file SHA256 is invalid")
    selected_ids = reserve_selection.get("selected_pair_ids")
    if (
        reserve_selection.get("status") != "PASS"
        or reserve_selection.get("requested_pair_count") != requested
        or reserve_selection.get("successful_pair_count") != requested
        or type(selected_ids) is not list
        or selected_ids != [row.get("pair_id") for row in labels]
    ):
        raise NeuroSEDFixedBudgetError("reserve selection does not bind label order")
    required = {
        "schema_version": "tastemolnet_neurosed_official_ged_label_v1",
        "status": "SUCCESS",
        "ged_method": "f2",
        "direction": OFFICIAL_GED_DIRECTION,
        "pyged_return_dtype": "float64",
        "label_dtype": "float32",
        "label_transform": "official_torch_float32_storage_cast_only",
        "target_representation": "ordered_query_target_lower_upper_interval",
        "bound_average_used": False,
        "single_bound_substitution_used": False,
        "approximate_or_neural_label_used": False,
        "cache_symmetric": False,
        "reverse_cache_shared": False,
        "query_target_order_in_cache_key": True,
    }
    pair_ids: list[str] = []
    seen_pair_ids: set[str] = set()
    exact_count = 0
    common_fields = (
        "gedlib_commit",
        "pyged_module_sha256",
        "gedlib_config_sha256",
        "feature_schema_sha256",
        "ged_method_args",
        "pair_sampler_manifest_sha256",
        "gedlib_build_manifest_sha256",
    )
    common: dict[str, Any] | None = None
    for index, raw in enumerate(labels):
        row = dict(raw)
        if any(row.get(key) != value for key, value in required.items()):
            raise NeuroSEDFixedBudgetError(f"official GED label row {index} changed")
        lower = _finite_nonnegative(row.get("lower_bound"), label="stored lower bound")
        upper = _finite_nonnegative(row.get("upper_bound"), label="stored upper bound")
        exact = row.get("exact_bound")
        if (
            lower > upper
            or type(exact) is not bool
            or row.get("stored_bounds_equal") is not (lower == upper)
            or row.get("bounds_kind") != ("exact" if exact else "interval")
        ):
            raise NeuroSEDFixedBudgetError("stored GED interval is invalid")
        if row.get("query_split") != split or row.get("target_split") != split:
            raise NeuroSEDFixedBudgetError("stored GED label split changed")
        if (
            not str(row.get("query_graph_id") or "")
            or not str(row.get("target_graph_id") or "")
            or row.get("query_graph_id") == row.get("target_graph_id")
        ):
            raise NeuroSEDFixedBudgetError("stored GED label graph roles changed")
        pair_id = str(row.get("pair_id") or "")
        if (
            len(pair_id) != 64
            or any(character not in "0123456789abcdef" for character in pair_id)
            or pair_id in seen_pair_ids
        ):
            raise NeuroSEDFixedBudgetError("official GED label pair IDs are invalid")
        pair_ids.append(pair_id)
        seen_pair_ids.add(pair_id)
        expected_cache_key = directional_ged_cache_key(
            query_canonical_graph_sha256=str(
                row.get("query_canonical_graph_sha256") or ""
            ),
            target_canonical_graph_sha256=str(
                row.get("target_canonical_graph_sha256") or ""
            ),
            gedlib_config_sha256=str(row.get("gedlib_config_sha256") or ""),
            feature_schema_sha256=str(row.get("feature_schema_sha256") or ""),
        )
        if row.get("cache_key") != expected_cache_key:
            raise NeuroSEDFixedBudgetError("official GED label cache key changed")
        exact_count += int(exact)
        observed_common = {key: row.get(key) for key in common_fields}
        if common is None:
            common = observed_common
        elif observed_common != common:
            raise NeuroSEDFixedBudgetError("GED backend changed within label file")
    assert common is not None
    attempted = reserve_selection.get("attempted_pair_count")
    timeout_count = reserve_selection.get("timeout_count")
    error_count = reserve_selection.get("error_count")
    reserve_used = reserve_selection.get("reserve_used")
    if (
        type(attempted) is not int
        or type(timeout_count) is not int
        or type(error_count) is not int
        or type(reserve_used) is not int
        or min(attempted, timeout_count, error_count, reserve_used) < 0
        or attempted != requested + timeout_count + error_count
        or reserve_used != attempted - requested
    ):
        raise NeuroSEDFixedBudgetError("reserve outcome accounting changed")
    payload = {
        "schema_version": PAIR_LABELS_MANIFEST_SCHEMA,
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "split": split,
        "requested_pair_count": requested,
        "attempted_pair_count": attempted,
        "successful_pair_count": requested,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "reserve_used": reserve_used,
        "pair_ids_sha256": _stable_sha256(pair_ids),
        "exact_bound_pair_count": exact_count,
        "interval_bound_pair_count": requested - exact_count,
        "finite_labels": True,
        "all_lower_bounds_le_upper_bounds": True,
        "real_pyged_gedlib_labels": True,
        "timeout_or_error_rows_used_as_labels": False,
        "selected_in_sampler_order": True,
        "ged_value_based_selection_used": False,
        "compact_storage_format": compact_storage_format,
        "compact_labels_sha256": str(compact_labels_sha256),
        "large_per_pair_json_debug_dump_used": False,
        "cache_symmetric": False,
        "reverse_cache_shared": False,
        "query_target_order_in_cache_key": True,
        "calibration_loaded": False,
        "test_loaded": False,
        **common,
    }
    payload["manifest_sha256"] = _stable_sha256(payload)
    return payload


__all__ = [
    "BENCHMARK_SCHEMA",
    "BENCHMARK_SUMMARY_SCHEMA",
    "GEDLIB_WORKER_RESOURCE_EVIDENCE_SCHEMA",
    "GEDLIB_WORKER_SELECTION_SCHEMA",
    "LEGAL_GEDLIB_WORKER_COUNTS",
    "NeuroSEDFixedBudgetError",
    "OFFICIAL_SED_EDIT_COSTS",
    "OFFICIAL_GED_DIRECTION",
    "PAIR_BUDGET_PLAN_SCHEMA",
    "PAIR_LABELS_MANIFEST_SCHEMA",
    "REVIEWED_WORKER_RESOURCE_EVIDENCE_PRODUCER_SHA256",
    "blocked_gedlib_worker_resource_evidence",
    "build_gedlib_worker_selection_manifest",
    "build_official_pair_labels_manifest",
    "combine_disjoint_benchmark_reports",
    "directional_ged_cache_key",
    "ged_cache_key",
    "ged_cache_symmetry_policy",
    "official_ged_interval_label",
    "select_fixed_pair_budget",
    "select_gedlib_worker_count",
    "select_successful_reserve_pairs",
    "summarize_real_gedlib_observations",
    "validate_benchmark_report",
    "validate_gedlib_worker_selection_manifest",
    "validation_pair_budget",
]
