"""Plain-text reporting helpers for evaluation summaries."""

from __future__ import annotations

from src.eval.interfaces import EvaluationSummary


def render_summary(summary: EvaluationSummary) -> str:
    """Format summary metrics for logs or terminal output."""

    lines = [f"example_count: {summary.example_count}"]
    for name in sorted(summary.metric_values):
        lines.append(f"{name}: {summary.metric_values[name]:.4f}")
    for note in summary.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
