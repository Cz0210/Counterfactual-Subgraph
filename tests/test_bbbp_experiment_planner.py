from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.eval.candidate_budget import nested_candidate_budget_prefixes
from src.eval.selector_ablation import SELECTOR_VARIANTS
from src.experiments.bbbp_framework import (
    ALL_PLAN_NAMES,
    load_plans,
    plans_json,
    render_future_shell,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_plans_validate_and_freeze_common_protocol() -> None:
    plans = load_plans("all", project_root=ROOT)
    assert tuple(plan.name for plan in plans) == ALL_PLAN_NAMES
    assert all(plan.protocol["cf_mode"] == "strict_flip" for plan in plans)
    assert all(plan.protocol["distance_line"] == "MolCLR-Node-Wasserstein" for plan in plans)
    assert all(plan.protocol["threshold_source"] == "calibration" for plan in plans)
    assert all(plan.protocol["test_usage"] == "final_evaluation_only" for plan in plans)
    assert all(plan.seeds == (0, 1, 2) for plan in plans)
    assert {stage.method for stage in plans[0].stages} >= {
        "ours",
        "globalgce",
        "gcfexplainer",
        "comrecgc",
    }


def test_plan_json_resources_are_not_fake_zeroes() -> None:
    payload = plans_json(load_plans("all", project_root=ROOT))
    assert payload["submission_performed"] is False
    for plan in payload["plans"]:
        for stage in plan["stages"]:
            assert stage["status"] == "NOT_RUN"
            assert all(value is None for value in stage["resources"].values())


def test_emit_shell_is_guarded_and_never_reports_submission() -> None:
    shell = render_future_shell(load_plans("common4", project_root=ROOT))
    assert "DO_NOT_EXECUTE_WITHOUT_SEPARATE_EXPERIMENT_AUTHORIZATION=1" in shell
    assert "scripts/exp_sbatch.sh" in shell
    assert "Submitted batch job" not in shell
    assert "sbatch " not in shell


def test_plan_cli_validate_only_does_not_submit_or_write_registry(tmp_path: Path) -> None:
    registry = ROOT / "outputs/hpc/experiment_registry/jobs.jsonl"
    before = registry.read_bytes() if registry.is_file() else None
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plan_bbbp_experiments.py"),
            "--plan",
            "all",
            "--validate-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Submitted batch job" not in completed.stdout + completed.stderr
    assert "submission_performed=false" in completed.stdout
    after = registry.read_bytes() if registry.is_file() else None
    assert after == before
    assert not list(tmp_path.iterdir())


def test_selector_registry_contains_requested_component_ablations() -> None:
    assert set(SELECTOR_VARIANTS) == {
        "full_selector",
        "no_cf_term",
        "no_coverage_term",
        "no_structural_redundancy",
        "no_coverage_redundancy",
        "no_size_penalty",
        "cfdrop_only",
        "coverage_only",
        "random_topk",
    }


def test_candidate_budget_prefixes_are_nested() -> None:
    rows = [
        {
            "parent_id": parent,
            "generation_rank": rank,
            "candidate_id": f"{parent}-{rank}",
        }
        for parent in ("p1", "p2")
        for rank in range(1, 9)
    ]
    prefixes = nested_candidate_budget_prefixes(rows)
    assert sorted(prefixes) == [1, 2, 4, 8]
    for parent in ("p1", "p2"):
        ids8 = [row["candidate_id"] for row in prefixes[8] if row["parent_id"] == parent]
        for budget in (1, 2, 4):
            ids = [row["candidate_id"] for row in prefixes[budget] if row["parent_id"] == parent]
            assert ids == ids8[:budget]


def test_configs_are_json_compatible_yaml_and_framework_only() -> None:
    for path in sorted((ROOT / "configs/experiments").glob("bbbp_*.yaml")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "FRAMEWORK_ONLY_NOT_RUN"
        assert payload["seeds"] == [0, 1, 2]
        assert payload["protocol"]["selection_performed_in_eval"] is False
        assert payload["protocol"]["threshold_fitted_on_test"] is False
