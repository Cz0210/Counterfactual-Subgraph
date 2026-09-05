"""One LLM prerequisite: independently replayed, temperature-corrected seed 7.

Never accept the historical hash-only package or a caller-provided PASS flag.
The GNN repair verifier owns every scientific check; this module only binds the
transport bytes and requires its exact corrected terminal state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ablations.llm.bace_native_runtime import verified_file


def require_corrected_gnn_core(archive_path: str | Path, archive_sha256: str) -> dict[str, Any]:
    from src.ablations.gnn.temperature_repair import verify_corrective_package

    archive = verified_file({"path": str(archive_path), "sha256": archive_sha256})
    audit = verify_corrective_package(archive, output_root=None)
    if audit.get("state") != "GNN_CORE_SEED7_CORRECTED_PASS":
        raise ValueError("WAITING_GNN_CORE_SEED7_CORRECTED_PASS")
    return {**audit, "verified_archive_sha256": archive_sha256,
            "main_matrix_count_required": False, "secondary_seeds_required": False,
            "gpu_borrow_enabled": False}
