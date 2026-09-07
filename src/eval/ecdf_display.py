"""Bounded display-only simplification of a frozen right-continuous ECDF.

No candidate selection, inference, integration or metric recomputation occurs.
Vertical jumps are represented explicitly. All numerical reporting must continue
to consume the original exact observations, never the simplified polyline.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def staircase(observations: Iterable[tuple[float, float]], keys=()) -> list[Point]:
    rows = sorted((float(x), float(y)) for x, y in observations)
    if not rows or len({x for x, _ in rows}) != len(rows):
        raise ValueError('Require nonempty unique exact thresholds')
    if any(not math.isfinite(x) or not math.isfinite(y) or x < 0 or not 0 <= y <= 1 for x, y in rows):
        raise ValueError('Invalid ECDF coordinate')
    if any(b[1] < a[1] for a, b in zip(rows, rows[1:])):
        raise ValueError('Exact ECDF must be monotone')
    for key in keys:
        key = float(key)
        if not rows[0][0] <= key <= rows[-1][0]:
            raise ValueError('Key threshold outside supplied full range')
        if key not in {x for x, _ in rows}:
            value = next(y for x, y in reversed(rows) if x < key)
            rows.append((key, value))
            rows.sort()
    result = [Point(*rows[0])]
    for x, y in rows[1:]:
        previous_y = result[-1].y
        result.append(Point(x, previous_y))
        if y != previous_y:
            result.append(Point(x, y))
    return result


def _errors(points: list[Point], lo: int, hi: int):
    a, b = points[lo], points[hi]
    if a.x == b.x:
        return [(i, 0.0 if a.y <= points[i].y <= b.y else math.inf) for i in range(lo + 1, hi)]
    return [(i, abs(points[i].y - (a.y + (b.y - a.y) * (points[i].x - a.x) / (b.x - a.x))))
            for i in range(lo + 1, hi)]


def simplify(points: list[Point], *, keys=(), max_error=0.005,
             plateau_fraction=0.02) -> tuple[list[Point], dict]:
    """Use the same vertical-error and long-plateau rule for every method.

    The error bound holds on both sides of every original ECDF jump, and over
    every intervening linear segment. The output preserves every source point
    whose omission violates the bound; no tolerance is tuned from rankings.
    """
    if not points or not 0 <= max_error <= 0.005 or not 0 <= plateau_fraction <= 1:
        raise ValueError('Invalid display policy')
    if any(b.x < a.x or b.y < a.y for a, b in zip(points, points[1:])):
        raise ValueError('Nonmonotone exact staircase')
    if any(not math.isfinite(p.x) or not math.isfinite(p.y) or not 0 <= p.y <= 1 for p in points):
        raise ValueError('Invalid staircase')
    forced = {0, len(points) - 1}
    key_set = {float(k) for k in keys}
    forced.update(i for i, p in enumerate(points) if p.x in key_set)
    if not key_set <= {p.x for p in points}:
        raise ValueError('Insert key thresholds into exact staircase first')
    width = points[-1].x - points[0].x
    for i, (a, b) in enumerate(zip(points, points[1:])):
        jump = a.x == b.x and b.y - a.y >= max_error
        plateau = a.y == b.y and (a.y == 0 or b.x - a.x >= width * plateau_fraction)
        if jump or plateau:
            forced.update((i, i + 1))
    selected = set(forced)
    initial = sorted(forced)
    pending = list(zip(initial, initial[1:]))
    while pending:
        lo, hi = pending.pop()
        errors = _errors(points, lo, hi)
        if not errors:
            continue
        index, error = max(errors, key=lambda pair: (pair[1], -pair[0]))
        if error > max_error:
            selected.add(index)
            pending.extend(((lo, index), (index, hi)))
    indices = sorted(selected)
    measured = max((error for lo, hi in zip(indices, indices[1:])
                    for _, error in _errors(points, lo, hi)), default=0.0)
    if measured > max_error:
        raise AssertionError('Display error exceeds frozen policy')
    return [points[i] for i in indices], {
        'schema_version': 'frozen_ecdf_display_v1',
        'state': 'PASS', 'source_points': len(points), 'display_points': len(indices),
        'error_unit': 'coverage_fraction', 'maximum_absolute_error': measured,
        'maximum_error_percentage_points': 100 * measured,
        'allowed_error_percentage_points': 100 * max_error,
        'key_thresholds': sorted(key_set), 'preserved_indices': sorted(forced),
        'plateau_min_range_fraction': plateau_fraction,
        'display_used_for_metrics_auc_or_ranking': False,
        'source_values_changed': False,
    }
