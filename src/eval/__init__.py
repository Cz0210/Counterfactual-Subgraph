"""Evaluation interfaces, metrics, and reporting helpers."""

from src.eval.candidate_pool_audit import (
    AuditConfig,
    audit_candidate_pool,
    render_audit_report,
    write_audit_outputs,
)
from src.eval.candidate_pool_merge import MergeConfig, merge_candidate_pools
from src.eval.class_counterfactual_selector import (
    SelectorConfig,
    select_class_counterfactual_subgraphs,
)
from src.eval.full_candidate_pool import (
    FullPoolGenerationConfig,
    generate_full_candidate_pool,
    inspect_checkpoint_directory,
    resolve_adapter_load_path,
)
from src.eval.full_candidate_pool_audit import (
    FullPoolAuditConfig,
    audit_full_candidate_pool,
)
from src.eval.inference import propose_fragment_candidate, run_minimal_inference
from src.eval.interfaces import EvaluationExample, EvaluationSummary, Evaluator
from src.eval.metrics import mean_metric, safe_rate
from src.eval.reporting import render_summary
from src.eval.selected_subgraph_overlap import compare_selected_subgraph_overlap

__all__ = [
    "AuditConfig",
    "EvaluationExample",
    "EvaluationSummary",
    "Evaluator",
    "FullPoolAuditConfig",
    "FullPoolGenerationConfig",
    "MergeConfig",
    "SelectorConfig",
    "audit_candidate_pool",
    "audit_full_candidate_pool",
    "generate_full_candidate_pool",
    "inspect_checkpoint_directory",
    "mean_metric",
    "merge_candidate_pools",
    "propose_fragment_candidate",
    "compare_selected_subgraph_overlap",
    "render_audit_report",
    "render_summary",
    "resolve_adapter_load_path",
    "run_minimal_inference",
    "safe_rate",
    "select_class_counterfactual_subgraphs",
    "write_audit_outputs",
]
