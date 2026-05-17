"""Evaluation interfaces, metrics, and reporting helpers."""

from src.eval.candidate_pool_audit import (
    AuditConfig,
    audit_candidate_pool,
    render_audit_report,
    write_audit_outputs,
)
from src.eval.inference import propose_fragment_candidate, run_minimal_inference
from src.eval.interfaces import EvaluationExample, EvaluationSummary, Evaluator
from src.eval.metrics import mean_metric, safe_rate
from src.eval.reporting import render_summary

__all__ = [
    "AuditConfig",
    "EvaluationExample",
    "EvaluationSummary",
    "Evaluator",
    "audit_candidate_pool",
    "mean_metric",
    "propose_fragment_candidate",
    "render_audit_report",
    "render_summary",
    "run_minimal_inference",
    "safe_rate",
    "write_audit_outputs",
]
