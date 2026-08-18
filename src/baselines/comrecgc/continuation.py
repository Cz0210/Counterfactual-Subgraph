"""Fail-closed decisions for a sliced COMRECGC generation continuation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint_audit import audit_generation_checkpoint


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit_complete_generation(
    generation_dir: str | Path, *, expected_steps: int
) -> dict[str, Any]:
    root = Path(generation_dir).expanduser().resolve()
    complete_path = root / "_RUN_COMPLETE.json"
    progress_path = root / "progress.json"
    manifest_path = root / "run_manifest.json"
    complete = _json(complete_path) if complete_path.is_file() else {}
    progress = _json(progress_path) if progress_path.is_file() else {}
    manifest = _json(manifest_path) if manifest_path.is_file() else {}
    current_step = int(progress.get("current_step") or 0)
    checks = {
        "run_complete_marker": complete.get("run_complete") is True,
        "progress_complete": current_step == int(expected_steps),
        "progress_run_complete": progress.get("run_complete") is True,
        "manifest_run_complete": manifest.get("run_complete") is True,
        "counterfactual_payload_present": (root / "counterfactuals.pt").is_file(),
    }
    return {
        "generation_dir": str(root),
        "expected_steps": int(expected_steps),
        "current_step": current_step,
        "checks": checks,
        "complete": all(checks.values()),
    }


def _checkpoint_roots(root: Path) -> list[Path]:
    values = [root]
    for pattern in ("checkpoints/*", "graph_state/checkpoints/*"):
        values.extend(path for path in root.glob(pattern) if path.is_dir())
    return sorted(set(values), key=lambda path: str(path))


def decide_resume_or_finalize(
    generation_dir: str | Path, *, expected_steps: int = 50_000
) -> dict[str, Any]:
    """Return ALREADY_COMPLETE, RESUME_SAFE, or FAIL_CLOSED.

    This function never interprets a nonempty output directory as a checkpoint.
    A continuation is permitted only by the strict RNG/state/closure manifest
    audited by :func:`audit_generation_checkpoint`.
    """

    root = Path(generation_dir).expanduser().resolve()
    complete = audit_complete_generation(root, expected_steps=expected_steps)
    if complete["complete"]:
        return {
            "schema_version": "comrecgc_resume_or_finalize_v1",
            "status": "ALREADY_COMPLETE",
            "generation_complete": True,
            "resume_safe": False,
            "fresh_start_allowed": False,
            "complete_audit": complete,
            "checkpoint_audits": [],
            "selected_checkpoint": None,
        }

    audits: list[dict[str, Any]] = []
    for checkpoint_root in _checkpoint_roots(root):
        audit = audit_generation_checkpoint(checkpoint_root)
        audit["candidate_root"] = str(checkpoint_root)
        audits.append(audit)
    safe = [audit for audit in audits if audit.get("RESUME_SAFE") is True]
    safe.sort(
        key=lambda audit: int(audit.get("checkpoint_current_step") or -1),
        reverse=True,
    )
    selected = safe[0] if safe else None
    return {
        "schema_version": "comrecgc_resume_or_finalize_v1",
        "status": "RESUME_SAFE" if selected else "FAIL_CLOSED",
        "generation_complete": False,
        "resume_safe": selected is not None,
        "fresh_start_allowed": False,
        "complete_audit": complete,
        "checkpoint_audits": audits,
        "selected_checkpoint": selected,
        "reason": (
            "strict_checkpoint_manifest_passed"
            if selected
            else "no_atomic_rng_transition_trace_closure_checkpoint"
        ),
    }
