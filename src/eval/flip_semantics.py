"""Shared teacher flip definitions for counterfactual evaluation."""

from __future__ import annotations

from typing import Any


TEACHER_STRICT_FLIP_DEFINITION = (
    "pred_before == target_label and pred_after != target_label"
)
OLD_WEAK_FLIP_DEFINITION = "pred_after != target_label"


def parse_label(value: Any) -> int | None:
    """Parse scalar labels emitted by CSV readers, NumPy, or torch models."""

    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def old_weak_flip(pred_after: Any, target_label: Any) -> bool:
    """Legacy audit-only condition that ignores the teacher's initial state."""

    after = parse_label(pred_after)
    target = parse_label(target_label)
    return bool(after is not None and target is not None and after != target)


def teacher_strict_flip(pred_before: Any, pred_after: Any, target_label: Any) -> bool:
    """Return true only for an actual teacher transition away from target_label."""

    before = parse_label(pred_before)
    target = parse_label(target_label)
    return bool(
        before is not None
        and target is not None
        and before == target
        and old_weak_flip(pred_after, target)
    )


def teacher_flip_audit_fields(
    pred_before: Any,
    pred_after: Any,
    target_label: Any,
) -> dict[str, Any]:
    """Build the canonical detail-row fields for strict and legacy flip audits."""

    strict = teacher_strict_flip(pred_before, pred_after, target_label)
    weak = old_weak_flip(pred_after, target_label)
    return {
        "cf_flip": strict,
        "teacher_strict_flip": strict,
        "old_weak_flip": weak,
        "flip_definition": TEACHER_STRICT_FLIP_DEFINITION,
    }
