"""One LLM prerequisite: independently replayed, temperature-corrected seed 7.

Never accept the historical hash-only package or a caller-provided PASS flag.
The GNN repair verifier owns every scientific check; this module only binds the
transport bytes and requires its exact corrected terminal state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from src.ablations.llm.bace_native_runtime import verified_file
from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file


def archive_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Corrective archive is not a physical regular file")
    stat = path.stat()
    return {"path": str(path.absolute()), "device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _corrective_proof_complete(proof):
    return (proof.get("state") == "GNN_CORE_SEED7_CORRECTED_PASS" and proof.get("seed") == 7
            and proof.get("validation_counts") == {name: 187 for name in ("gin", "gcn", "gatv2", "gatedgcn_plus")}
            and proof.get("counts") == {"calibration": 288, "test": 614}
            and all(proof.get(k) is True for k in ("all_weights_unchanged", "gine_unchanged", "candidate_pool_unchanged", "selectors_frozen_before_test", "native_common_metrics_replayed"))
            and proof.get("raw_ot_recomputed_count") == 0 and proof.get("cache_provenance_gaps") == []
            and proof.get("main_matrix_write") is False and proof.get("repair_selected_using_test") is False
            and all(isinstance(proof.get(k), str) and len(proof[k]) == 64
                    for k in ("independent_science_replay_sha256", "original_package_sha256", "repair_contract_sha256")))


def adopt_existing_acceptance(*, archive_path, archive_sha256, overlay, audit, output_path):
    """First owner adoption of an already independently verified local import.

    Verify two small receipts plus the outer archive once. Never unpack/replay
    it again. This is not acceptance of an unauthenticated PASS string.
    """
    output = Path(output_path)
    if output.exists():
        raise ValueError("Owner acceptance output must be fresh")
    imported = json.loads(verified_file(overlay).read_text())
    proof = json.loads(verified_file(audit).read_text())
    if (proof.get("self_sha256") != canonical_json_sha256({k: v for k, v in proof.items() if k != "self_sha256"})
            or imported.get("corrective_audit_sha256") != audit["sha256"]
            or any(imported.get(k) != v for k, v in proof.items())
            or imported.get("package_sha256") != archive_sha256
            or imported.get("sha256") != archive_sha256
            or Path(imported.get("source_archive", "")) != Path(archive_path)
            or not _corrective_proof_complete(proof)):
        raise ValueError("EXISTING_CORRECTIVE_ACCEPTANCE_CHAIN_MISMATCH")
    archive = verified_file({"path": str(archive_path), "sha256": archive_sha256})
    identity = archive_identity(archive)
    if identity["size"] != imported.get("bytes"):
        raise ValueError("Imported archive byte count changed")
    atomic_json(output, {"schema_version": "bace_llm_corrected_core_acceptance_v1",
                "archive_identity": identity, "archive_sha256": archive_sha256,
                "source_import_overlay": overlay, "source_independent_audit": audit,
                "independent_audit": proof, "scientific_replay_performed_here": False})
    return {"path": str(output.absolute()), "sha256": sha256_file(output)}


def require_corrected_gnn_core(archive_path: str | Path, archive_sha256: str, *,
                               acceptance: dict[str, str] | None = None) -> dict[str, Any]:
    # An acceptance is sealed only after the independent corrective verifier.
    # Reuse that proof while the local archive identity is unchanged; no model
    # hashes, archive unpacking or scientific replay occur in the resource loop.
    if acceptance is not None:
        receipt = json.loads(verified_file(acceptance).read_text())
        if (receipt.get("schema_version") != "bace_llm_corrected_core_acceptance_v1"
                or receipt.get("archive_sha256") != archive_sha256
                or receipt.get("archive_identity") != archive_identity(archive_path)
                or not _corrective_proof_complete(receipt.get("independent_audit", {}))):
            raise ValueError("CORRECTED_ACCEPTANCE_CHANGED_REVERIFY_FIRST_ADOPTION")
        return receipt["independent_audit"]
    from src.ablations.gnn.temperature_repair import verify_corrective_package

    archive = verified_file({"path": str(archive_path), "sha256": archive_sha256})
    audit = verify_corrective_package(archive, output_root=None)
    if audit.get("state") != "GNN_CORE_SEED7_CORRECTED_PASS":
        raise ValueError("WAITING_GNN_CORE_SEED7_CORRECTED_PASS")
    return {**audit, "verified_archive_sha256": archive_sha256,
            "main_matrix_count_required": False, "secondary_seeds_required": False,
            "gpu_borrow_enabled": False}
