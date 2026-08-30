"""Frozen 20k/25k resource-cap decision for BACE ComRecGC.

This module is intentionally read-only.  It converts one already validated
2,500-step convergence audit into a durable scheduling recommendation; it has
no process signalling or post-processing API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class BaceComRecGCResourceCapError(RuntimeError):
    """The committed audit cannot support a resource-cap decision."""


@dataclass(frozen=True, slots=True)
class ResourceCapPolicy:
    first_cap_step: int = 20_000
    absolute_cap_step: int = 25_000
    check_interval: int = 2_500
    minimum_valid_unique_count: int = 10


POLICY = ResourceCapPolicy()


def decide_bace_comrecgc_resource_cap(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the explicitly authorized cap without reading cal/test data."""

    status = str(audit.get("status") or "")
    if status not in {"CONTINUE", "CONVERGED_EARLY_STOP"}:
        return {
            "status": "WAIT",
            "reason": f"audit_status={status or 'MISSING'}",
            "policy": asdict(POLICY),
            "signals_sent": False,
            "postprocess_started": False,
        }
    summaries = audit.get("checkpoint_summaries")
    checkpoints = audit.get("checkpoint_evidence")
    if not isinstance(summaries, list) or not summaries:
        raise BaceComRecGCResourceCapError("audit has no checkpoint summaries")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise BaceComRecGCResourceCapError("audit has no checkpoint evidence")
    latest = summaries[-1]
    checkpoint = checkpoints[-1]
    if not isinstance(latest, Mapping) or not isinstance(checkpoint, Mapping):
        raise BaceComRecGCResourceCapError("latest committed evidence is malformed")
    step = latest.get("step")
    unique = latest.get("valid_unique_count")
    lineage = latest.get("lineage_error_count")
    for label, value in (
        ("step", step),
        ("valid_unique_count", unique),
        ("lineage_error_count", lineage),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaceComRecGCResourceCapError(f"{label} is invalid")
    if (
        audit.get("evaluation_step") != step
        or checkpoint.get("step") != step
        or step % POLICY.check_interval != 0
    ):
        raise BaceComRecGCResourceCapError("audit/checkpoint boundary differs")
    digest = checkpoint.get("checkpoint_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise BaceComRecGCResourceCapError("checkpoint digest is invalid")
    common = {
        "policy": asdict(POLICY),
        "m_effective": step,
        "valid_unique_count": unique,
        "lineage_error_count": lineage,
        "checkpoint_digest": digest,
        "checkpoint_evidence": dict(checkpoint),
        "test_loaded": False,
        "signals_sent": False,
        "postprocess_started": False,
    }
    if status == "CONVERGED_EARLY_STOP":
        return {
            "status": "HANDOVER_ELIGIBLE",
            "reason": "PREREGISTERED_CONVERGENCE_PASS",
            **common,
        }
    if step < POLICY.first_cap_step:
        return {"status": "CONTINUE", "reason": "BELOW_FIRST_CAP", **common}
    eligible = lineage == 0 and unique >= POLICY.minimum_valid_unique_count
    if eligible:
        return {
            "status": "HANDOVER_ELIGIBLE",
            "reason": (
                "RESOURCE_CAP_20000"
                if step == POLICY.first_cap_step
                else "RESOURCE_CAP_25000_FALLBACK"
            ),
            **common,
        }
    if step < POLICY.absolute_cap_step:
        return {
            "status": "CONTINUE",
            "reason": "EXTEND_TO_ABSOLUTE_CAP_25000",
            **common,
        }
    return {
        "status": "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
        "reason": "ABSOLUTE_CAP_INSUFFICIENT_RULES_OR_LINEAGE_ERRORS",
        **common,
    }


__all__ = [
    "BaceComRecGCResourceCapError",
    "POLICY",
    "ResourceCapPolicy",
    "decide_bace_comrecgc_resource_cap",
]
