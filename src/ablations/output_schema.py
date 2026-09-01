"""Output inventories and non-numeric templates for ablation runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LLM_VARIANT_OUTPUT_FILES = (
    "candidate_pool.jsonl",
    "candidate_metrics.json",
    "candidate_metrics.csv",
    "verification_manifest.json",
    "selector_manifest.json",
    "selected_rules.json",
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_k10.csv",
    "novelty_metrics.json",
    "run_manifest.json",
    "final_audit.json",
)

LLM_AGGREGATE_OUTPUT_FILES = (
    "llm_ablation_candidate_quality.csv",
    "llm_ablation_main_metrics.csv",
    "llm_ablation_table.tex",
    "llm_ablation_figure3.pdf",
    "llm_ablation_manifest.json",
)

GNN_VARIANT_OUTPUT_FILES = (
    "classifier_metrics.json",
    "calibration_metrics.json",
    "native_cohort_manifest.json",
    "common_cohort_manifest.json",
    "verification_manifest.json",
    "selected_rules.json",
    "explanation_metrics_native.json",
    "explanation_metrics_common.json",
    "rule_overlap_with_gine.json",
    "covered_parent_overlap.json",
    "run_manifest.json",
    "final_audit.json",
)

GNN_AGGREGATE_OUTPUT_FILES = (
    "gnn_ablation_classifier_table.csv",
    "gnn_ablation_explanation_table.csv",
    "gnn_ablation_table.tex",
    "gnn_rule_stability.csv",
    "gnn_ablation_manifest.json",
)


def output_inventory(family: str, *, aggregate: bool = False) -> tuple[str, ...]:
    key = str(family).strip().lower()
    if key == "llm":
        return LLM_AGGREGATE_OUTPUT_FILES if aggregate else LLM_VARIANT_OUTPUT_FILES
    if key == "gnn":
        return GNN_AGGREGATE_OUTPUT_FILES if aggregate else GNN_VARIANT_OUTPUT_FILES
    raise KeyError(f"unknown ablation output family: {family}")


def run_manifest_template(family: str) -> dict[str, Any]:
    """Return a schema-only manifest; no result value is fabricated."""

    if family not in {"llm", "gnn"}:
        raise KeyError(family)
    payload = {
        "schema_version": f"{family}_ablation_run_manifest_v1",
        "status": "CONFIG_ONLY",
        "science_started": False,
        "main_matrix_authority_mutated": False,
        "contract": None,
        "artifacts": {name: None for name in output_inventory(family)},
        "metrics": None,
    }
    return deepcopy(payload)


__all__ = [
    "GNN_AGGREGATE_OUTPUT_FILES",
    "GNN_VARIANT_OUTPUT_FILES",
    "LLM_AGGREGATE_OUTPUT_FILES",
    "LLM_VARIANT_OUTPUT_FILES",
    "output_inventory",
    "run_manifest_template",
]
