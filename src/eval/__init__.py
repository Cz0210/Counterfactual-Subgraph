"""Evaluation interfaces, metrics, and reporting helpers."""

from src.eval.inference import propose_fragment_candidate, run_minimal_inference
from src.eval.interfaces import EvaluationExample, EvaluationSummary, Evaluator
from src.eval.metrics import mean_metric, safe_rate
from src.eval.reporting import render_summary

__all__ = [
    "EvaluationExample",
    "EvaluationSummary",
    "Evaluator",
    "mean_metric",
    "propose_fragment_candidate",
    "render_summary",
    "run_minimal_inference",
    "safe_rate",
]
