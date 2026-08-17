"""Strict loader for method-shared, calibration-frozen WNode thresholds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_values(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("thresholds")
    if isinstance(raw, list) and raw:
        return [float(value) for value in raw]
    rows = payload.get("raw_quantile_thresholds")
    if isinstance(rows, list) and rows:
        return [float(row["threshold"]) for row in rows if isinstance(row, dict)]
    return []


def load_shared_frozen_thresholds(path: str | Path) -> dict[str, Any]:
    """Load thresholds without fitting, fallback, or method-specific adaptation."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Frozen threshold manifest must be an object: {source}")
    thresholds = _threshold_values(payload)
    if not thresholds or any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("Frozen WNode thresholds must be finite and nonnegative.")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("Frozen WNode thresholds must be unique and increasing.")
    theta_star = float(payload["theta_star"])
    if not any(math.isclose(theta_star, value, rel_tol=0.0, abs_tol=1e-12) for value in thresholds):
        raise ValueError("Frozen theta_star is absent from the threshold grid.")
    if payload.get("threshold_fitted_on_test") is not False:
        raise ValueError("Frozen WNode threshold must explicitly be test-independent.")
    for field in ("selection_used_test", "test_used_for_selection"):
        if payload.get(field) is True:
            raise ValueError(f"Frozen threshold manifest reports test leakage: {field}")
    if payload.get("shared_across_methods") is not True:
        raise ValueError("Frozen WNode threshold must be shared across methods.")
    cf_mode = str(payload.get("cf_mode") or "strict_flip")
    if cf_mode != "strict_flip":
        raise ValueError(f"Frozen threshold cf_mode must be strict_flip, got {cf_mode!r}.")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "thresholds": thresholds,
        "threshold_csv": ",".join(format(value, ".17g") for value in thresholds),
        "theta_star": theta_star,
        "cost_cap": float(payload.get("cost_cap", thresholds[-1])),
        "threshold_source": str(payload.get("threshold_source") or ""),
        "threshold_fitted_on_test": False,
        "shared_across_methods": True,
        "cf_mode": cf_mode,
        "action_semantics_version": payload.get("action_semantics_version"),
    }
