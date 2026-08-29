"""Selection contract for the bounded non-MIP Taste GEDLIB route."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.utils.tastemolnet_neurosed_gedlib_build import (
    GED_LABEL_BACKEND_VARIANT,
    NON_MIP_METHOD_CONFIGS,
)


SELECTION_SCHEMA = "tastemolnet_neurosed_non_mip_gedlib_selection_v1"
PRIMARY_TRAIN_PAIRS = 5_000
PRIMARY_VALIDATION_PAIRS = 1_000
REDUCED_TRAIN_PAIRS = 2_000
REDUCED_VALIDATION_PAIRS = 500
PAIR_SAMPLING_SEED = 7


class NonMIPGEDLIBSelectionError(RuntimeError):
    """A candidate or fixed-budget decision violates the release contract."""


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, *, label: str, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise NonMIPGEDLIBSelectionError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NonMIPGEDLIBSelectionError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise NonMIPGEDLIBSelectionError(f"{label} must be finite")
    return parsed


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise NonMIPGEDLIBSelectionError(f"{label} must be a lowercase SHA256")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_report(
    *,
    method: str,
    method_args: str,
    pair_ids: Sequence[str],
    replay_observations: Sequence[Sequence[Mapping[str, Any]]],
    replay_wall_seconds: Sequence[float],
    replay_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate two fixed-input replays and summarize one candidate."""

    expected = list(pair_ids)
    if (
        method not in NON_MIP_METHOD_CONFIGS
        or method_args != NON_MIP_METHOD_CONFIGS[method]
        or len(expected) != 100
        or len(set(expected)) != 100
        or len(replay_observations) != 2
        or len(replay_wall_seconds) != 2
        or len(replay_artifacts) != 2
    ):
        raise NonMIPGEDLIBSelectionError("candidate replay authority changed")
    normalized_replays: list[list[dict[str, Any]]] = []
    replay_evidence: list[dict[str, Any]] = []
    successes: list[int] = []
    for replay_index, rows in enumerate(replay_observations):
        if len(rows) != 100:
            raise NonMIPGEDLIBSelectionError("candidate replay must contain 100 pairs")
        normalized: list[dict[str, Any]] = []
        success_count = 0
        for row_index, (pair_id, raw) in enumerate(zip(expected, rows)):
            row = dict(raw)
            if row.get("pair_id") != pair_id:
                raise NonMIPGEDLIBSelectionError("candidate replay pair order changed")
            status = row.get("status")
            if status == "SUCCESS":
                lower = _finite(row.get("lower_bound"), label="lower_bound")
                upper = _finite(row.get("upper_bound"), label="upper_bound")
                if lower > upper:
                    raise NonMIPGEDLIBSelectionError("GEDLIB lower bound exceeds upper")
                success_count += 1
                outcome = {
                    "pair_id": pair_id,
                    "status": status,
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            elif status in {"TIMEOUT", "GEDLIB_ERROR"}:
                if row.get("lower_bound") is not None or row.get("upper_bound") is not None:
                    raise NonMIPGEDLIBSelectionError(
                        "failed GEDLIB row retained a bound"
                    )
                outcome = {
                    "pair_id": pair_id,
                    "status": status,
                    "lower_bound": None,
                    "upper_bound": None,
                }
            else:
                raise NonMIPGEDLIBSelectionError(
                    f"candidate replay row {replay_index}:{row_index} is malformed"
                )
            normalized.append(outcome)
        normalized_replays.append(normalized)
        successes.append(success_count)
    walls = [
        _finite(value, label="candidate replay wall seconds")
        for value in replay_wall_seconds
    ]
    if any(value <= 0 or value > 600 for value in walls) or sum(walls) > 600:
        raise NonMIPGEDLIBSelectionError("candidate exceeded its ten-minute wall")
    pair_ids_sha256 = _stable_sha256(expected)
    for replay_index, (artifact_raw, normalized, success_count, wall) in enumerate(
        zip(replay_artifacts, normalized_replays, successes, walls), start=1
    ):
        artifact = dict(artifact_raw)
        observations_path = Path(str(artifact.get("observations_path") or ""))
        benchmark_report_path = Path(str(artifact.get("benchmark_report_path") or ""))
        if not observations_path.is_absolute() or not benchmark_report_path.is_absolute():
            raise NonMIPGEDLIBSelectionError("replay artifact paths must be absolute")
        outcome_sha256 = _stable_sha256(normalized)
        if (
            artifact.get("replay_index") != replay_index
            or artifact.get("method") != method
            or artifact.get("method_args") != method_args
            or artifact.get("pair_ids_sha256") != pair_ids_sha256
            or artifact.get("outcome_sha256") != outcome_sha256
            or artifact.get("successful_pair_count") != success_count
            or not math.isclose(
                _finite(
                    artifact.get("selector_observed_wall_seconds"),
                    label="selector observed replay wall",
                ),
                wall,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise NonMIPGEDLIBSelectionError("replay artifact binding changed")
        evidence = {
            "replay_index": replay_index,
            "method": method,
            "method_args": method_args,
            "observations_path": str(observations_path),
            "observations_sha256": _sha256(
                artifact.get("observations_sha256"), label="observations SHA256"
            ),
            "benchmark_report_path": str(benchmark_report_path),
            "benchmark_report_sha256": _sha256(
                artifact.get("benchmark_report_sha256"),
                label="benchmark report SHA256",
            ),
            "benchmark_status": str(artifact.get("benchmark_status") or ""),
            "pair_ids_sha256": pair_ids_sha256,
            "outcome_sha256": outcome_sha256,
            "successful_pair_count": success_count,
            "selector_observed_wall_seconds": wall,
            "benchmark_wall_seconds": _finite(
                artifact.get("benchmark_wall_seconds"),
                label="benchmark wall seconds",
            ),
            "pyged_module_sha256": _sha256(
                artifact.get("pyged_module_sha256"), label="pyged module SHA256"
            ),
            "gedlib_commit": str(artifact.get("gedlib_commit") or ""),
        }
        if evidence["benchmark_status"] not in {"PASS", "PASS_WITH_GEDLIB_ERRORS"}:
            raise NonMIPGEDLIBSelectionError("benchmark replay status is invalid")
        commit = evidence["gedlib_commit"]
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise NonMIPGEDLIBSelectionError("replay GEDLIB commit is invalid")
        replay_evidence.append(evidence)
    if replay_evidence[0]["pyged_module_sha256"] != replay_evidence[1]["pyged_module_sha256"]:
        raise NonMIPGEDLIBSelectionError("pyged module changed between replays")
    if replay_evidence[0]["gedlib_commit"] != replay_evidence[1]["gedlib_commit"]:
        raise NonMIPGEDLIBSelectionError("GEDLIB commit changed between replays")
    deterministic = normalized_replays[0] == normalized_replays[1]
    minimum_success_rate = min(successes) / 100.0
    mean_wall = mean(walls)
    mean_successes = mean(successes)
    report: dict[str, Any] = {
        "method": method,
        "method_args": method_args,
        "candidate_is_pinned_gedlib_method": True,
        "ged_backend_variant": "non_mip",
        "GED_LABEL_BACKEND_VARIANT": GED_LABEL_BACKEND_VARIANT,
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
        "ged_label_backend_variant": GED_LABEL_BACKEND_VARIANT,
        "f2_blp_used": False,
        "gurobi_used": False,
        "pair_count_per_replay": 100,
        "replay_count": 2,
        "same_fixed_pair_cohort_replayed": True,
        "pair_ids_sha256": pair_ids_sha256,
        "pyged_module_sha256": replay_evidence[0]["pyged_module_sha256"],
        "gedlib_commit": replay_evidence[0]["gedlib_commit"],
        "replays": replay_evidence,
        "successful_pair_counts": successes,
        "minimum_success_rate": minimum_success_rate,
        "finite_lower_le_upper": True,
        "deterministic_replay": deterministic,
        "replay_wall_seconds": walls,
        "candidate_total_wall_seconds": sum(walls),
        "mean_wall_seconds_per_100_pairs": mean_wall,
        "mean_successful_pairs_per_hour": mean_successes * 3600.0 / mean_wall,
        "eligible": deterministic and minimum_success_rate >= 0.95,
    }
    report["report_sha256"] = _stable_sha256(report)
    return report


def select_non_mip_backend(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    worker_count: int,
    pyged_module_sha256: str,
    gedlib_commit: str,
) -> dict[str, Any]:
    """Select the fastest passing candidate and fix the 5k/1k or 2k/500 budget."""

    if (
        not 1 <= len(reports) <= len(NON_MIP_METHOD_CONFIGS)
        or set(reports) - set(NON_MIP_METHOD_CONFIGS)
    ):
        raise NonMIPGEDLIBSelectionError(
            "only authenticated deterministic non-MIP candidates are allowed"
        )
    if type(worker_count) is not int or worker_count not in (1, 2, 4, 8):
        raise NonMIPGEDLIBSelectionError("selected worker count is invalid")
    pyged_module_sha256 = _sha256(
        pyged_module_sha256, label="selected pyged module SHA256"
    )
    if len(gedlib_commit) != 40 or any(
        character not in "0123456789abcdef" for character in gedlib_commit
    ):
        raise NonMIPGEDLIBSelectionError("selected GEDLIB commit is invalid")
    checked: dict[str, dict[str, Any]] = {}
    pair_hash: str | None = None
    total_wall = 0.0
    eligible: list[tuple[float, str]] = []
    for method, raw in reports.items():
        report = dict(raw)
        recorded_sha = report.pop("report_sha256", None)
        if recorded_sha != _stable_sha256(report):
            raise NonMIPGEDLIBSelectionError("candidate report hash changed")
        report["report_sha256"] = recorded_sha
        if (
            report.get("method") != method
            or report.get("method_args") != NON_MIP_METHOD_CONFIGS[method]
            or report.get("ged_label_backend_variant") != GED_LABEL_BACKEND_VARIANT
            or report.get("GED_LABEL_BACKEND_VARIANT") != GED_LABEL_BACKEND_VARIANT
            or report.get("F2_BLP_USED") is not False
            or report.get("GUROBI_USED") is not False
            or report.get("f2_blp_used") is not False
            or report.get("gurobi_used") is not False
        ):
            raise NonMIPGEDLIBSelectionError("candidate backend identity changed")
        replays = report.get("replays")
        if type(replays) is not list or len(replays) != 2:
            raise NonMIPGEDLIBSelectionError("candidate replay evidence is missing")
        success_counts = [replay.get("successful_pair_count") for replay in replays]
        replay_walls = [
            _finite(
                replay.get("selector_observed_wall_seconds"),
                label="selector observed replay wall",
            )
            for replay in replays
        ]
        if any(type(value) is not int or not 0 <= value <= 100 for value in success_counts):
            raise NonMIPGEDLIBSelectionError("candidate success accounting changed")
        recomputed_minimum_success_rate = min(success_counts) / 100.0
        recomputed_deterministic = (
            replays[0].get("outcome_sha256") == replays[1].get("outcome_sha256")
        )
        recomputed_mean_wall = mean(replay_walls)
        recomputed_throughput = mean(success_counts) * 3600.0 / recomputed_mean_wall
        recomputed_total_wall = sum(replay_walls)
        recomputed_eligible = (
            recomputed_deterministic and recomputed_minimum_success_rate >= 0.95
        )
        if (
            report.get("successful_pair_counts") != success_counts
            or not math.isclose(
                _finite(report.get("minimum_success_rate"), label="success rate"),
                recomputed_minimum_success_rate,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or report.get("deterministic_replay") is not recomputed_deterministic
            or report.get("eligible") is not recomputed_eligible
            or not math.isclose(
                _finite(report.get("candidate_total_wall_seconds"), label="total wall"),
                recomputed_total_wall,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _finite(report.get("mean_wall_seconds_per_100_pairs"), label="mean wall"),
                recomputed_mean_wall,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _finite(
                    report.get("mean_successful_pairs_per_hour"),
                    label="successful throughput",
                ),
                recomputed_throughput,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise NonMIPGEDLIBSelectionError(
                "candidate derived eligibility/throughput changed"
            )
        if (
            report.get("pyged_module_sha256") != pyged_module_sha256
            or report.get("gedlib_commit") != gedlib_commit
            or any(
                replay.get("pyged_module_sha256") != pyged_module_sha256
                or replay.get("gedlib_commit") != gedlib_commit
                for replay in replays
            )
        ):
            raise NonMIPGEDLIBSelectionError("candidate backend binary changed")
        if pair_hash is None:
            pair_hash = str(report["pair_ids_sha256"])
        elif report.get("pair_ids_sha256") != pair_hash:
            raise NonMIPGEDLIBSelectionError("candidate pair cohorts differ")
        wall = recomputed_total_wall
        total_wall += wall
        if recomputed_eligible:
            eligible.append(
                (
                    -_finite(
                        report.get("mean_successful_pairs_per_hour"),
                        label="candidate successful throughput",
                    ),
                    method,
                )
            )
        checked[method] = report
    if total_wall > 1800:
        raise NonMIPGEDLIBSelectionError("backend selection exceeded thirty minutes")
    if not eligible:
        raise NonMIPGEDLIBSelectionError("no non-MIP GEDLIB candidate passed")
    selected_method = min(eligible)[1]
    selected = checked[selected_method]
    pairs_per_hour = _finite(
        selected["mean_successful_pairs_per_hour"], label="selected throughput"
    )
    if pairs_per_hour <= 0:
        raise NonMIPGEDLIBSelectionError("selected throughput must be positive")
    projected_primary_hours = (
        PRIMARY_TRAIN_PAIRS + PRIMARY_VALIDATION_PAIRS
    ) / pairs_per_hour
    reduced = projected_primary_hours > 24.0
    train_pairs = REDUCED_TRAIN_PAIRS if reduced else PRIMARY_TRAIN_PAIRS
    validation_pairs = (
        REDUCED_VALIDATION_PAIRS if reduced else PRIMARY_VALIDATION_PAIRS
    )
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "status": "PASS",
        "marker": "[TASTE_NON_MIP_GEDLIB_BACKEND_SELECTED]",
        "GED_LABEL_BACKEND_VARIANT": GED_LABEL_BACKEND_VARIANT,
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
        "ged_label_backend_variant": GED_LABEL_BACKEND_VARIANT,
        "ged_backend_variant": "non_mip",
        "selected_ged_backend": selected_method,
        "backend_config": selected["method_args"],
        "f2_blp_used": False,
        "gurobi_used": False,
        "original_f2_blp_configuration_claimed": False,
        "real_pinned_gedlib_used": True,
        "pyged_module_sha256": pyged_module_sha256,
        "gedlib_commit": gedlib_commit,
        "candidate_count": len(checked),
        "candidate_selection_total_wall_seconds": total_wall,
        "selection_policy": (
            "highest_eligible_successful_pairs_per_hour_then_method_name"
        ),
        "label_contract": {
            "representation": "lower_upper_interval_bounds",
            "direction": "query_to_target",
            "finite_required": True,
            "lower_le_upper_required": True,
            "pyged_return_dtype": "float64",
            "bound_average_used": False,
            "official_pair_builder_consumable": True,
        },
        "minimum_success_rate_required": 0.95,
        "fixed_seed_replay_required": True,
        "candidate_reports": checked,
        "selected_worker_count": worker_count,
        "pair_sampling_seed": PAIR_SAMPLING_SEED,
        "projected_primary_6000_pair_label_hours": projected_primary_hours,
        "maximum_primary_label_hours": 24.0,
        "fixed_pair_budget": True,
        "resource_reduced_budget": reduced,
        "selected_neurosed_train_pair_budget": train_pairs,
        "selected_neurosed_validation_pair_budget": validation_pairs,
        "official_pair_semantics": True,
        "official_model_training_semantics": True,
        "independent_query_target_pairs": True,
        "parent_own_subgraph_shortcut": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "gcf_runtime_direction": "generated_query_to_original_target",
    }
    payload["selection_sha256"] = _stable_sha256(payload)
    return payload


def validate_non_mip_selection_manifest(
    manifest: Mapping[str, Any],
    *,
    reopen_artifacts: bool,
) -> dict[str, Any]:
    """Recompute selection, optionally reopening every benchmark artifact."""

    payload = dict(manifest)
    recorded_sha256 = _sha256(
        payload.pop("selection_sha256", None), label="selection SHA256"
    )
    if recorded_sha256 != _stable_sha256(payload):
        raise NonMIPGEDLIBSelectionError("selection manifest hash changed")
    payload["selection_sha256"] = recorded_sha256
    reports_raw = payload.get("candidate_reports")
    if type(reports_raw) is not dict:
        raise NonMIPGEDLIBSelectionError("selection candidate reports are missing")
    reports = {str(method): dict(report) for method, report in reports_raw.items()}
    if reopen_artifacts:
        from src.eval.tastemolnet_neurosed_fixed_budget import (
            validate_benchmark_report,
        )

        rebuilt_reports: dict[str, dict[str, Any]] = {}
        for method, report in reports.items():
            replay_evidence = report.get("replays")
            if type(replay_evidence) is not list or len(replay_evidence) != 2:
                raise NonMIPGEDLIBSelectionError("candidate replay evidence is missing")
            observations_by_replay: list[list[dict[str, Any]]] = []
            pair_ids: list[str] | None = None
            for replay in replay_evidence:
                observations_path = Path(str(replay.get("observations_path") or ""))
                benchmark_path = Path(str(replay.get("benchmark_report_path") or ""))
                for path, expected_sha, label in (
                    (
                        observations_path,
                        replay.get("observations_sha256"),
                        "observations",
                    ),
                    (
                        benchmark_path,
                        replay.get("benchmark_report_sha256"),
                        "benchmark report",
                    ),
                ):
                    if (
                        not path.is_absolute()
                        or not path.is_file()
                        or path.is_symlink()
                        or _sha256_file(path) != expected_sha
                    ):
                        raise NonMIPGEDLIBSelectionError(
                            f"{label} artifact changed after selection"
                        )
                rows: list[dict[str, Any]] = []
                with observations_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            value = json.loads(line)
                            if type(value) is not dict:
                                raise NonMIPGEDLIBSelectionError(
                                    "observation artifact row is malformed"
                                )
                            rows.append(value)
                benchmark_value = json.loads(benchmark_path.read_text(encoding="utf-8"))
                if type(benchmark_value) is not dict:
                    raise NonMIPGEDLIBSelectionError("benchmark artifact is malformed")
                benchmark = validate_benchmark_report(
                    benchmark_value, expected_budget=100
                )
                if (
                    benchmark.get("ged_method") != method
                    or benchmark.get("ged_method_args")
                    != NON_MIP_METHOD_CONFIGS.get(method)
                    or benchmark.get("pyged_module_sha256")
                    != payload.get("pyged_module_sha256")
                    or benchmark.get("gedlib_commit") != payload.get("gedlib_commit")
                ):
                    raise NonMIPGEDLIBSelectionError(
                        "benchmark artifact backend changed"
                    )
                if pair_ids is None:
                    pair_ids = list(benchmark["pair_ids"])
                elif benchmark.get("pair_ids") != pair_ids:
                    raise NonMIPGEDLIBSelectionError(
                        "replay benchmark pair cohort changed"
                    )
                observations_by_replay.append(rows)
            assert pair_ids is not None
            rebuilt_reports[method] = build_candidate_report(
                method=method,
                method_args=NON_MIP_METHOD_CONFIGS[method],
                pair_ids=pair_ids,
                replay_observations=observations_by_replay,
                replay_wall_seconds=[
                    float(replay["selector_observed_wall_seconds"])
                    for replay in replay_evidence
                ],
                replay_artifacts=replay_evidence,
            )
            if rebuilt_reports[method] != report:
                raise NonMIPGEDLIBSelectionError(
                    "candidate report differs from reopened artifacts"
                )
        reports = rebuilt_reports
    recomputed = select_non_mip_backend(
        reports,
        worker_count=payload.get("selected_worker_count"),
        pyged_module_sha256=str(payload.get("pyged_module_sha256") or ""),
        gedlib_commit=str(payload.get("gedlib_commit") or ""),
    )
    for key, value in recomputed.items():
        if key != "selection_sha256" and payload.get(key) != value:
            raise NonMIPGEDLIBSelectionError(f"selection field changed: {key}")
    for path_field, sha_field in (
        ("build_manifest_path", "build_manifest_sha256"),
        ("pair_sampler_manifest_path", "pair_sampler_manifest_sha256"),
        ("pairs_jsonl_path", "pairs_jsonl_sha256"),
        ("graph_inventory_path", "graph_inventory_sha256"),
    ):
        if path_field in payload:
            path = Path(str(payload[path_field]))
            if (
                not path.is_absolute()
                or not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != payload.get(sha_field)
            ):
                raise NonMIPGEDLIBSelectionError(
                    f"selection authority changed: {path_field}"
                )
    return payload


__all__ = [
    "NonMIPGEDLIBSelectionError",
    "PAIR_SAMPLING_SEED",
    "PRIMARY_TRAIN_PAIRS",
    "PRIMARY_VALIDATION_PAIRS",
    "REDUCED_TRAIN_PAIRS",
    "REDUCED_VALIDATION_PAIRS",
    "SELECTION_SCHEMA",
    "build_candidate_report",
    "select_non_mip_backend",
    "validate_non_mip_selection_manifest",
]
