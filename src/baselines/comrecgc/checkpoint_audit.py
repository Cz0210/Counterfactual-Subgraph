"""Strict referential-closure audit for COMRECGC generation checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import sha256_file
from .live_graph_state import AuthoritativeGraphStore


CHECKPOINT_MANIFEST_NAMES = (
    "generation_checkpoint_manifest.json",
    "checkpoint_manifest.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit_generation_checkpoint(generation_dir: str | Path) -> dict[str, Any]:
    root = Path(generation_dir).expanduser().resolve()
    progress_path = root / "progress.json"
    progress = _load_json(progress_path) if progress_path.is_file() else {}
    candidates = [
        path
        for name in CHECKPOINT_MANIFEST_NAMES
        for path in (root / name, root / "graph_state" / name)
        if path.is_file()
    ]
    inventory = sorted(
        str(path)
        for pattern in ("*checkpoint*", "graph_state/*checkpoint*")
        for path in root.glob(pattern)
        if path.is_file()
    )
    result: dict[str, Any] = {
        "schema_version": "comrecgc_generation_checkpoint_audit_v1",
        "generation_dir": str(root),
        "checkpoint_inventory": inventory,
        "checkpoint_manifest_found": len(candidates) == 1,
        "checkpoint_manifest_path": str(candidates[0]) if len(candidates) == 1 else None,
        "progress_path": str(progress_path),
        "progress_current_step": progress.get("current_step"),
        "RESUME_SAFE": False,
        "reasons": [],
    }
    if len(candidates) != 1:
        result["reasons"].append(
            "missing_checkpoint_manifest" if not candidates else "ambiguous_checkpoint_manifests"
        )
        return result

    manifest_path = candidates[0]
    manifest = _load_json(manifest_path)
    result["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    rng_keys = set((manifest.get("rng_state") or {}).keys())
    current_hashes = [str(value) for value in manifest.get("current_graph_hashes") or []]
    transition_sources = [str(value) for value in manifest.get("transition_source_hashes") or []]
    transition_destinations = [
        str(value) for value in manifest.get("transition_destination_hashes") or []
    ]
    live_hashes = [str(value) for value in manifest.get("live_reference_hashes") or []]
    resolvable = {str(value) for value in manifest.get("resolvable_hashes") or []}
    closure = set(current_hashes) | set(transition_sources) | set(transition_destinations) | set(live_hashes)
    checks = {
        "atomic_complete": manifest.get("atomic_complete") is True,
        "step_matches_progress": manifest.get("current_step") == progress.get("current_step"),
        "rng_state_complete": required_rng.issubset(rng_keys),
        "current_graph_hashes_present": bool(current_hashes),
        "referential_closure_resolvable": closure.issubset(resolvable),
        "unresolved_lookup_count_zero": int(manifest.get("unresolved_lookups", -1)) == 0,
    }
    store_path_value = manifest.get("backing_store_path")
    store_audit: dict[str, Any] | None = None
    if store_path_value:
        store_path = Path(store_path_value)
        if not store_path.is_absolute():
            store_path = (manifest_path.parent / store_path).resolve()
        if store_path.is_file():
            store = AuthoritativeGraphStore(store_path, read_only=True)
            try:
                store_audit = store.integrity_audit()
            finally:
                store.close()
            checks["backing_store_integrity"] = bool(store_audit["integrity_passed"])
            checks["backing_store_checksum_matches"] = (
                store_audit["content_sha256"]
                == manifest.get("backing_store_content_sha256")
            )
        else:
            checks["backing_store_integrity"] = False
            checks["backing_store_checksum_matches"] = False
    else:
        checks["backing_store_integrity"] = False
        checks["backing_store_checksum_matches"] = False
    result.update(
        {
            "checkpoint_current_step": manifest.get("current_step"),
            "closure_hash_count": len(closure),
            "resolvable_hash_count": len(resolvable),
            "checks": checks,
            "backing_store_audit": store_audit,
        }
    )
    result["reasons"] = [name for name, passed in checks.items() if not passed]
    result["RESUME_SAFE"] = all(checks.values())
    return result
