from __future__ import annotations

import pytest

from src.eval.tastemolnet_neurosed_non_mip import (
    NonMIPGEDLIBSelectionError,
    build_candidate_report,
    select_non_mip_backend,
)
from src.utils.tastemolnet_neurosed_gedlib_build import NON_MIP_METHOD_CONFIGS


PAIR_IDS = [f"pair-{index:03d}" for index in range(100)]


def _stable(value) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rows(*, failures: int = 0, drift: bool = False):
    rows = []
    for index, pair_id in enumerate(PAIR_IDS):
        if index < failures:
            rows.append(
                {
                    "pair_id": pair_id,
                    "status": "GEDLIB_ERROR",
                    "lower_bound": None,
                    "upper_bound": None,
                }
            )
        else:
            rows.append(
                {
                    "pair_id": pair_id,
                    "status": "SUCCESS",
                    "lower_bound": 1.0,
                    "upper_bound": 2.0 + int(drift and index == 99),
                }
            )
    return rows


def _report(method: str, wall: float, *, failures: int = 0):
    rows = _rows(failures=failures)
    normalized = [
        {
            "pair_id": row["pair_id"],
            "status": row["status"],
            "lower_bound": row["lower_bound"],
            "upper_bound": row["upper_bound"],
        }
        for row in rows
    ]
    replay_artifacts = [
        {
            "replay_index": replay,
            "method": method,
            "method_args": NON_MIP_METHOD_CONFIGS[method],
            "observations_path": f"/tmp/{method}-replay-{replay}.jsonl",
            "observations_sha256": str(replay) * 64,
            "benchmark_report_path": f"/tmp/{method}-replay-{replay}.json",
            "benchmark_report_sha256": str(replay + 2) * 64,
            "benchmark_status": "PASS" if failures == 0 else "PASS_WITH_GEDLIB_ERRORS",
            "pair_ids_sha256": _stable(PAIR_IDS),
            "outcome_sha256": _stable(normalized),
            "successful_pair_count": 100 - failures,
            "selector_observed_wall_seconds": wall / 2,
            "benchmark_wall_seconds": wall / 2,
            "pyged_module_sha256": "a" * 64,
            "gedlib_commit": "b" * 40,
        }
        for replay in (1, 2)
    ]
    return build_candidate_report(
        method=method,
        method_args=NON_MIP_METHOD_CONFIGS[method],
        pair_ids=PAIR_IDS,
        replay_observations=(rows, rows),
        replay_wall_seconds=(wall / 2, wall / 2),
        replay_artifacts=replay_artifacts,
    )


def test_selects_eligible_non_mip_branch_backend_and_primary_budget() -> None:
    reports = {"branch": _report("branch", 20.0)}

    selected = select_non_mip_backend(
        reports,
        worker_count=8,
        pyged_module_sha256="a" * 64,
        gedlib_commit="b" * 40,
    )

    assert selected["selected_ged_backend"] == "branch"
    assert selected["GED_LABEL_BACKEND_VARIANT"] == "NON_MIP_GEDLIB"
    assert selected["F2_BLP_USED"] is False
    assert selected["GUROBI_USED"] is False
    assert selected["f2_blp_used"] is False
    assert selected["gurobi_used"] is False
    assert selected["selected_neurosed_train_pair_budget"] == 5_000
    assert selected["selected_neurosed_validation_pair_budget"] == 1_000


def test_selector_recomputes_throughput_instead_of_trusting_report_field() -> None:
    report = _report("branch", 600.0, failures=5)
    report["mean_successful_pairs_per_hour"] = 200.0
    report.pop("report_sha256")
    import hashlib
    import json

    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(NonMIPGEDLIBSelectionError, match="derived"):
        select_non_mip_backend(
            {"branch": report},
            worker_count=1,
            pyged_module_sha256="a" * 64,
            gedlib_commit="b" * 40,
        )


def test_determinism_drift_or_below_95_percent_is_ineligible() -> None:
    drift = build_candidate_report(
        method="branch",
        method_args=NON_MIP_METHOD_CONFIGS["branch"],
        pair_ids=PAIR_IDS,
        replay_observations=(_rows(), _rows(drift=True)),
        replay_wall_seconds=(10.0, 10.0),
        replay_artifacts=(
            {
                "replay_index": 1,
                "method": "branch",
                "method_args": NON_MIP_METHOD_CONFIGS["branch"],
                "observations_path": "/tmp/branch-drift-1.jsonl",
                "observations_sha256": "1" * 64,
                "benchmark_report_path": "/tmp/branch-drift-1.json",
                "benchmark_report_sha256": "3" * 64,
                "benchmark_status": "PASS",
                "pair_ids_sha256": _stable(PAIR_IDS),
                "outcome_sha256": _stable(
                    [
                        {
                            "pair_id": row["pair_id"],
                            "status": row["status"],
                            "lower_bound": row["lower_bound"],
                            "upper_bound": row["upper_bound"],
                        }
                        for row in _rows()
                    ]
                ),
                "successful_pair_count": 100,
                "selector_observed_wall_seconds": 10.0,
                "benchmark_wall_seconds": 10.0,
                "pyged_module_sha256": "a" * 64,
                "gedlib_commit": "b" * 40,
            },
            {
                "replay_index": 2,
                "method": "branch",
                "method_args": NON_MIP_METHOD_CONFIGS["branch"],
                "observations_path": "/tmp/branch-drift-2.jsonl",
                "observations_sha256": "2" * 64,
                "benchmark_report_path": "/tmp/branch-drift-2.json",
                "benchmark_report_sha256": "4" * 64,
                "benchmark_status": "PASS",
                "pair_ids_sha256": _stable(PAIR_IDS),
                "outcome_sha256": _stable(
                    [
                        {
                            "pair_id": row["pair_id"],
                            "status": row["status"],
                            "lower_bound": row["lower_bound"],
                            "upper_bound": row["upper_bound"],
                        }
                        for row in _rows(drift=True)
                    ]
                ),
                "successful_pair_count": 100,
                "selector_observed_wall_seconds": 10.0,
                "benchmark_wall_seconds": 10.0,
                "pyged_module_sha256": "a" * 64,
                "gedlib_commit": "b" * 40,
            },
        ),
    )
    with pytest.raises(NonMIPGEDLIBSelectionError, match="no non-MIP"):
        select_non_mip_backend(
            {"branch": drift},
            worker_count=1,
            pyged_module_sha256="a" * 64,
            gedlib_commit="b" * 40,
        )
