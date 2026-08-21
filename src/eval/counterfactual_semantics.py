"""Dataset-agnostic binary and multiclass counterfactual semantics.

All strict flips are transitions away from the declared source class.  The
implementation never derives a binary destination with ``1 - label`` and is
therefore shared by BACE (two classes) and TasteMolNet (three classes).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import operator
from typing import Any, Iterable, Mapping, Sequence


def _class_index(value: Any, *, name: str, num_classes: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer class index, not bool.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer class index: {value!r}.") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative: {normalized}.")
    if num_classes is not None and normalized >= num_classes:
        raise ValueError(
            f"{name}={normalized} is outside [0, {num_classes - 1}]."
        )
    return int(normalized)


def _probability_vector(
    values: Sequence[float],
    *,
    name: str,
    expected_classes: int | None = None,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric probability vector.") from exc
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two classes.")
    if expected_classes is not None and len(result) != expected_classes:
        raise ValueError(
            f"{name} has {len(result)} classes; expected {expected_classes}."
        )
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} contains a non-finite or out-of-range probability.")
    total = sum(result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to one; observed {total:.12g}.")
    return result


def strict_flip(pred_before: Any, pred_after: Any, source_label: Any) -> bool:
    """Return whether the classifier moved away from its source class."""

    source = _class_index(source_label, name="source_label")
    before = _class_index(pred_before, name="pred_before")
    after = _class_index(pred_after, name="pred_after")
    return before == source and after != source


def multiclass_flip(pred_before: Any, pred_after: Any, source_label: Any) -> bool:
    """Explicit multiclass alias used by tests and downstream adapters."""

    return strict_flip(pred_before, pred_after, source_label)


def counterfactual_drop(
    probabilities_before: Sequence[float],
    probabilities_after: Sequence[float],
    source_label: Any,
) -> float:
    """Return the drop in probability assigned to the source class."""

    before = _probability_vector(probabilities_before, name="probabilities_before")
    after = _probability_vector(
        probabilities_after,
        name="probabilities_after",
        expected_classes=len(before),
    )
    source = _class_index(
        source_label,
        name="source_label",
        num_classes=len(before),
    )
    return float(before[source] - after[source])


def source_class_margin(probabilities: Sequence[float], source_label: Any) -> float:
    """Return source probability minus the strongest competing probability."""

    values = _probability_vector(probabilities, name="probabilities")
    source = _class_index(
        source_label,
        name="source_label",
        num_classes=len(values),
    )
    strongest_other = max(
        probability
        for class_index, probability in enumerate(values)
        if class_index != source
    )
    return float(values[source] - strongest_other)


@dataclass(frozen=True, slots=True)
class CounterfactualRecord:
    """One auditable classifier transition before and after intervention."""

    source_label: int
    pred_before: int
    pred_after: int
    destination_label: int
    p_before_all_classes: tuple[float, ...]
    p_after_all_classes: tuple[float, ...]
    source_prob_before: float
    source_prob_after: float
    cf_drop: float
    margin_before: float
    margin_after: float
    margin_drop: float
    cf_flip: bool
    rule_id: str | None = None

    @property
    def num_classes(self) -> int:
        return len(self.p_before_all_classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "pred_before": self.pred_before,
            "pred_after": self.pred_after,
            "destination_label": self.destination_label,
            "p_before_all_classes": list(self.p_before_all_classes),
            "p_after_all_classes": list(self.p_after_all_classes),
            "source_prob_before": self.source_prob_before,
            "source_prob_after": self.source_prob_after,
            "cf_drop": self.cf_drop,
            "margin_before": self.margin_before,
            "margin_after": self.margin_after,
            "margin_drop": self.margin_drop,
            "cf_flip": self.cf_flip,
            "rule_id": self.rule_id,
        }


def compute_counterfactual_semantics(
    *,
    source_label: Any,
    pred_before: Any,
    pred_after: Any,
    probabilities_before: Sequence[float],
    probabilities_after: Sequence[float],
    rule_id: str | None = None,
) -> CounterfactualRecord:
    """Build the complete binary/multiclass intervention record."""

    before_probabilities = _probability_vector(
        probabilities_before,
        name="probabilities_before",
    )
    after_probabilities = _probability_vector(
        probabilities_after,
        name="probabilities_after",
        expected_classes=len(before_probabilities),
    )
    num_classes = len(before_probabilities)
    source = _class_index(
        source_label,
        name="source_label",
        num_classes=num_classes,
    )
    before = _class_index(pred_before, name="pred_before", num_classes=num_classes)
    after = _class_index(pred_after, name="pred_after", num_classes=num_classes)
    margin_before = source_class_margin(before_probabilities, source)
    margin_after = source_class_margin(after_probabilities, source)
    source_probability_before = before_probabilities[source]
    source_probability_after = after_probabilities[source]
    return CounterfactualRecord(
        source_label=source,
        pred_before=before,
        pred_after=after,
        destination_label=after,
        p_before_all_classes=before_probabilities,
        p_after_all_classes=after_probabilities,
        source_prob_before=source_probability_before,
        source_prob_after=source_probability_after,
        cf_drop=float(source_probability_before - source_probability_after),
        margin_before=margin_before,
        margin_after=margin_after,
        margin_drop=float(margin_before - margin_after),
        cf_flip=before == source and after != source,
        rule_id=None if rule_id is None else str(rule_id),
    )


def _row_fields(row: CounterfactualRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, CounterfactualRecord):
        return row.to_dict()
    if not isinstance(row, Mapping):
        raise TypeError(
            "Destination-distribution rows must be CounterfactualRecord or mapping."
        )
    return dict(row)


def _one_distribution(
    rows: list[dict[str, Any]],
    *,
    source_label: int,
    num_classes: int,
    label_map: Mapping[int, str] | None,
) -> dict[str, Any]:
    counts = {destination: 0 for destination in range(num_classes) if destination != source_label}
    strict_count = 0
    for row in rows:
        before = _class_index(
            row.get("pred_before"),
            name="pred_before",
            num_classes=num_classes,
        )
        after = _class_index(
            row.get("pred_after"),
            name="pred_after",
            num_classes=num_classes,
        )
        row_source = _class_index(
            row.get("source_label", source_label),
            name="source_label",
            num_classes=num_classes,
        )
        if row_source != source_label:
            raise ValueError(
                "Destination-distribution rows contain inconsistent source labels: "
                f"expected {source_label}, observed {row_source}."
            )
        computed_flip = before == source_label and after != source_label
        if "cf_flip" in row and bool(row["cf_flip"]) != computed_flip:
            raise ValueError("Recorded cf_flip disagrees with classifier transition.")
        if computed_flip:
            counts[after] += 1
            strict_count += 1

    transitions: dict[str, dict[str, Any]] = {}
    for destination, count in sorted(counts.items()):
        key = f"{source_label}->{destination}"
        transitions[key] = {
            "source_label": source_label,
            "source_label_name": (
                label_map.get(source_label) if label_map is not None else None
            ),
            "destination_label": destination,
            "destination_label_name": (
                label_map.get(destination) if label_map is not None else None
            ),
            "count": count,
            "rate": float(count / strict_count) if strict_count else 0.0,
        }
    return {
        "total_records": len(rows),
        "total_strict_flips": strict_count,
        "transitions": transitions,
    }


def destination_distribution(
    rows: Iterable[CounterfactualRecord | Mapping[str, Any]],
    *,
    source_label: Any,
    num_classes: int,
    label_map: Mapping[int, str] | None = None,
    rule_id_field: str = "rule_id",
) -> dict[str, Any]:
    """Summarize strict-flip destinations overall and for each selected rule."""

    if isinstance(num_classes, bool) or int(num_classes) < 2:
        raise ValueError("num_classes must be an integer greater than or equal to two.")
    class_count = int(num_classes)
    source = _class_index(source_label, name="source_label", num_classes=class_count)
    normalized_rows = [_row_fields(row) for row in rows]
    overall = _one_distribution(
        normalized_rows,
        source_label=source,
        num_classes=class_count,
        label_map=label_map,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        rule_id = row.get(rule_id_field)
        if rule_id is not None and str(rule_id).strip():
            grouped[str(rule_id)].append(row)
    return {
        "schema_version": 1,
        "source_label": source,
        "num_classes": class_count,
        "overall": overall,
        "by_rule": {
            rule_id: _one_distribution(
                grouped[rule_id],
                source_label=source,
                num_classes=class_count,
                label_map=label_map,
            )
            for rule_id in sorted(grouped)
        },
    }


__all__ = [
    "CounterfactualRecord",
    "compute_counterfactual_semantics",
    "counterfactual_drop",
    "destination_distribution",
    "multiclass_flip",
    "source_class_margin",
    "strict_flip",
]
