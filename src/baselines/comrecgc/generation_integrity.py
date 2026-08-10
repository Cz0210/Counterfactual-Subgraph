"""Post-generation integrity gate for the COMRECGC live graph state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint_audit import audit_generation_checkpoint
from .contracts import sha256_file
from .frozen_payload import payload_graphs_by_official_hash


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit_generation_integrity(
    generation_dir: str | Path, *, expected_steps: int = 50_000
) -> dict[str, Any]:
    root = Path(generation_dir).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    progress_path = root / "progress.json"
    complete_path = root / "_RUN_COMPLETE.json"
    graph_audit_path = root / "graph_state_audit.json"
    closure_audit_path = root / "frozen_payload_closure_audit.json"
    required = (
        manifest_path,
        progress_path,
        complete_path,
        graph_audit_path,
        closure_audit_path,
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Generation integrity inputs missing: {missing}")
    manifest = _json(manifest_path)
    progress = _json(progress_path)
    graph_audit = _json(graph_audit_path)
    closure_audit = _json(closure_audit_path)
    result_path = Path(manifest["counterfactuals_path"]).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Generation counterfactual artifact missing: {result_path}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC dependency
        raise RuntimeError("Generation integrity gate requires torch.") from exc
    try:
        payload = torch.load(result_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - pinned older torch
        payload = torch.load(result_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Generation counterfactual payload must be a dictionary.")
    graph_map = payload.get("graph_map")
    candidates = payload.get("counterfactual_candidates")
    if not isinstance(graph_map, dict) or not isinstance(candidates, list):
        raise TypeError("Generation payload graph_map/counterfactual_candidates are invalid.")
    selected_hashes = [row.get("graph_hash") for row in candidates if isinstance(row, dict)]
    frozen_graphs = payload_graphs_by_official_hash(payload)

    checkpoint = audit_generation_checkpoint(root)
    incomplete_checkpoint = bool(checkpoint["checkpoint_manifest_found"]) and not bool(
        checkpoint["RESUME_SAFE"]
    )
    checks = {
        "manifest_complete": manifest.get("run_complete") is True,
        "completed_steps_exact": int(progress.get("current_step", -1)) == int(expected_steps),
        "max_steps_exact": int(progress.get("max_steps", -1)) == int(expected_steps),
        "progress_complete": progress.get("run_complete") is True,
        "unresolved_lookups_zero": int(graph_audit.get("unresolved_lookups", -1)) == 0,
        "transition_sources_resolvable": int(
            graph_audit.get("unresolved_transition_source_count", -1)
        )
        == 0,
        "transition_destinations_valid": int(
            graph_audit.get("invalid_transition_destination_count", -1)
        )
        == 0,
        "selected_graph_hashes_resolvable": all(
            str(value) in frozen_graphs for value in selected_hashes
        ),
        "frozen_payload_closure_complete": closure_audit.get("closure_complete") is True,
        "frozen_payload_post_write_verified": closure_audit.get(
            "post_write_reload_verified"
        )
        is True,
        "frozen_payload_checksum_matches": closure_audit.get("payload_checksum")
        == sha256_file(result_path),
        "backing_store_integrity": bool(
            (graph_audit.get("backing_store") or {}).get("integrity_passed")
        ),
        "no_incomplete_checkpoint": not incomplete_checkpoint,
        "counterfactual_sha256_matches": sha256_file(result_path)
        == manifest.get("counterfactuals_sha256"),
    }
    return {
        "schema_version": "comrecgc_generation_integrity_gate_v1",
        "generation_dir": str(root),
        "expected_steps": int(expected_steps),
        "completed_steps": progress.get("current_step"),
        "candidate_count": len(candidates),
        "selected_graph_hash_count": len(selected_hashes),
        "transition_source_count": graph_audit.get("transition_source_count"),
        "transition_destination_count": graph_audit.get("transition_destination_count"),
        "unresolved_lookups": graph_audit.get("unresolved_lookups"),
        "backing_store_checksum_pass": bool(
            (graph_audit.get("backing_store") or {}).get("integrity_passed")
        ),
        "checkpoint_audit": checkpoint,
        "frozen_payload_closure_audit": closure_audit,
        "checks": checks,
        "generation_integrity_passed": all(checks.values()),
    }
