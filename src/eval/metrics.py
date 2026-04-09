"""Small metric helpers shared across evaluation code."""

from __future__ import annotations

from collections.abc import Iterable


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a bounded rate even when the denominator is zero."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def mean_metric(values: Iterable[float]) -> float:
    """Return a defensive mean for possibly empty sequences."""

    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(values_tuple) / len(values_tuple)
