"""Parent-level bootstrap summaries for molecular recourse metrics."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_METRICS = (
    "coverage",
    "cost",
    "cf_drop",
    "flip_rate",
    "valid_rate",
    "structural_redundancy",
    "coverage_redundancy",
)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    position = (len(ordered) - 1) * float(probability)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _group_sort_key(values: tuple[str, ...]) -> tuple[tuple[int, float | str], ...]:
    result: list[tuple[int, float | str]] = []
    for value in values:
        try:
            result.append((0, float(value)))
        except ValueError:
            result.append((1, value))
    return tuple(result)


def _parent_metric_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_field: str,
    metrics: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        parent_id = str(row.get(parent_field) or "").strip()
        if not parent_id:
            raise ValueError(f"Bootstrap row lacks {parent_field!r}.")
        grouped[parent_id].append(row)
    result: dict[str, dict[str, float | None]] = {}
    for parent_id, parent_rows in grouped.items():
        if len(parent_rows) != 1:
            raise ValueError(
                "Parent-level bootstrap requires one aggregated row per parent; "
                f"parent={parent_id!r} rows={len(parent_rows)}."
            )
        row = parent_rows[0]
        parsed: dict[str, float | None] = {}
        for metric in metrics:
            raw = row.get(metric)
            if raw is None or str(raw).strip() == "":
                parsed[metric] = None
                continue
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Bootstrap metric {metric} is non-finite.")
            parsed[metric] = value
        result[parent_id] = parsed
    return result


def parent_level_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_field: str = "parent_id",
    metrics: Sequence[str] = DEFAULT_METRICS,
    num_samples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    if int(num_samples) <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    by_parent = _parent_metric_values(
        rows,
        parent_field=parent_field,
        metrics=metrics,
    )
    parent_ids = sorted(by_parent)
    if len(parent_ids) < 2:
        raise ValueError("Parent-level bootstrap requires at least two parents.")
    rng = random.Random(int(seed))
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _sample in range(int(num_samples)):
        sampled = [parent_ids[rng.randrange(len(parent_ids))] for _ in parent_ids]
        for metric in metrics:
            values = [
                value
                for parent_id in sampled
                if (value := by_parent[parent_id][metric]) is not None
            ]
            if values:
                draws[metric].append(float(statistics.fmean(values)))
    summaries: dict[str, Any] = {}
    for metric, values in draws.items():
        if not values:
            summaries[metric] = {
                "mean": None,
                "std": None,
                "median": None,
                "ci_2_5": None,
                "ci_97_5": None,
                "num_bootstrap_values": 0,
            }
            continue
        summaries[metric] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "median": float(statistics.median(values)),
            "ci_2_5": float(_quantile(values, 0.025)),
            "ci_97_5": float(_quantile(values, 0.975)),
            "num_bootstrap_values": len(values),
        }
    return {
        "schema_version": "parent_level_bootstrap_v1",
        "resampling_unit": "parent_id",
        "pair_row_bootstrap": False,
        "num_parents": len(parent_ids),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "metrics": summaries,
    }


def parent_level_curve_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_field: str,
    parent_field: str = "parent_id",
    num_samples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    """Build pointwise confidence bands without treating pair rows as samples.

    Input must contain one value per parent and curve coordinate. Every curve
    coordinate must use the same parent universe; this prevents a method from
    obtaining a narrower band by silently dropping difficult parents.
    """

    if not group_fields:
        raise ValueError("Curve bootstrap requires at least one group field.")
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key_parts: list[str] = []
        for field in group_fields:
            value = str(row.get(field) or "").strip()
            if not value:
                raise ValueError(f"Curve bootstrap row lacks {field!r}.")
            key_parts.append(value)
        grouped[tuple(key_parts)].append(row)
    if not grouped:
        raise ValueError("Curve bootstrap rows are empty.")

    expected_parent_ids: tuple[str, ...] | None = None
    output_rows: list[dict[str, Any]] = []
    for point_index, key in enumerate(sorted(grouped, key=_group_sort_key)):
        point_rows = grouped[key]
        parent_ids = tuple(
            sorted(str(row.get(parent_field) or "").strip() for row in point_rows)
        )
        if any(not value for value in parent_ids):
            raise ValueError(f"Curve bootstrap row lacks {parent_field!r}.")
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError(
                "Curve bootstrap requires one row per parent and coordinate; "
                f"coordinate={key!r}."
            )
        if expected_parent_ids is None:
            expected_parent_ids = parent_ids
        elif parent_ids != expected_parent_ids:
            raise ValueError(
                "Curve bootstrap coordinates have different parent universes; "
                f"coordinate={key!r}."
            )

        summary = parent_level_bootstrap(
            point_rows,
            parent_field=parent_field,
            metrics=(value_field,),
            num_samples=num_samples,
            seed=int(seed) + point_index,
        )["metrics"][value_field]
        output = {field: value for field, value in zip(group_fields, key, strict=True)}
        output.update(
            {
                f"{value_field}_mean": summary["mean"],
                f"{value_field}_std": summary["std"],
                f"{value_field}_median": summary["median"],
                f"{value_field}_ci_2_5": summary["ci_2_5"],
                f"{value_field}_ci_97_5": summary["ci_97_5"],
                "num_parents": len(parent_ids),
                "num_bootstrap_values": summary["num_bootstrap_values"],
            }
        )
        output_rows.append(output)

    return {
        "schema_version": "parent_level_curve_confidence_band_v1",
        "resampling_unit": parent_field,
        "pair_row_bootstrap": False,
        "group_fields": list(group_fields),
        "value_field": value_field,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "num_parents": len(expected_parent_ids or ()),
        "rows": output_rows,
    }


__all__ = [
    "DEFAULT_METRICS",
    "parent_level_bootstrap",
    "parent_level_curve_bootstrap",
]
