"""Read-only preflight for the conditional Mut trace-off closeout.

This is deliberately not an executor. The pinned trace-off producer does not
seal the causal edit history required by the existing chemistry-repair stage.
A valid A/B scientific failure authorizes considering Route B, not bypassing
that missing input or launching a generation-only dead end.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

from src.utils.autodl_mut_first_divergence_v1 import file_sha256
from src.utils.autodl_mut_route_b_v1 import (
    CANDIDATE_CAPACITY,
    M_MAX,
    validate_route_b_evidence,
    validate_route_b_spec,
)


SCHEMA = "mut_route_b_closeout_preflight_v1"
DEFAULT_MIN_FREE_INODES = 100_000
PRODUCERS = {
    "generation": "scripts/autodl/run_mut_route_b_owner_v1.py",
    "generation_payload_closure": "src/baselines/comrecgc/runtime.py",
    "pair_and_exact_recourse": "scripts/baselines/comrecgc/run_common_recourse.py",
    "pair_universe_binding": "src/baselines/comrecgc/recourse.py",
    "compact_pair_store": "src/baselines/comrecgc/external_memory_recourse.py",
    "generation_checkpoint": "src/baselines/comrecgc/generation_checkpoint.py",
    "chemistry": "src/baselines/comrecgc/mutagenicity_chemistry_audit.py",
    "unified_eval": "scripts/baselines/comrecgc/run_slot_unified_eval.py",
    "final_freeze": "scripts/baselines/comrecgc/freeze_recovery_result.py",
    "canonical_publisher": "src/utils/autodl_mut_successor_stages_v1.py",
    "matrix_terminal_validator": "src/eval/non_taste_matrix_append.py",
}


def compact_resource_estimate(
    *, free_inodes: int, free_bytes: int, vector_dimension: int | None = None,
    vector_itemsize: int | None = None,
) -> dict[str, Any]:
    """Upper-bound known compact files; never replace the live admission gate.

    The new universe is not known before generation. Use capacity, not the old
    universe's observed pair count. Pair chunks retain two npy files each;
    consolidation temporarily coexists with them. Unmeasured generation/model,
    chemistry and evaluation byte peaks remain explicit unknowns.
    """
    if any(type(v) is not int or v < 0 for v in (free_inodes, free_bytes)):
        raise ValueError("filesystem availability must be non-negative integers")
    if (vector_dimension is None) != (vector_itemsize is None):
        raise ValueError("both measured vector dimension and dtype bytes are required")
    if vector_dimension is not None and (
        type(vector_dimension) is not int or vector_dimension <= 0
        or type(vector_itemsize) is not int or vector_itemsize not in (4, 8)
    ):
        raise ValueError("invalid measured recourse-vector shape/dtype")
    chunks = math.ceil(CANDIDATE_CAPACITY / 128)
    # Two stores, keep-last=2 plus one in-flight publication per store. Eight
    # inodes/bundle conservatively covers payload, SQLite snapshot, manifests,
    # completion/mirror receipts and the directory itself.
    checkpoint_peak_inodes = 2 * (2 + 1) * 8
    retention_receipt_allowance = 2 * math.ceil(M_MAX / 500)
    known_peak = 2 * chunks + checkpoint_peak_inodes + retention_receipt_allowance + 256
    pair_rows = CANDIDATE_CAPACITY * 1448
    pair_bytes = None if vector_dimension is None else pair_rows * (
        2 * 8 + vector_dimension * int(vector_itemsize)
    )
    return {
        "status": "ESTIMATE_ONLY_NOT_FULL_ADMISSION",
        "free_inodes": free_inodes,
        "free_bytes": free_bytes,
        "existing_min_free_inodes": DEFAULT_MIN_FREE_INODES,
        "existing_guard_shortfall_inodes": max(0, DEFAULT_MIN_FREE_INODES - free_inodes),
        "guard_modified": False,
        "candidate_count_upper_bound": CANDIDATE_CAPACITY,
        "parent_count": 1448,
        "pair_chunk_candidate_batch": 128,
        "pair_chunk_count_upper_bound": chunks,
        "pair_chunk_inodes": 2 * chunks,
        "mirrored_checkpoint_peak_inodes": checkpoint_peak_inodes,
        "retention_receipt_allowance_inodes": retention_receipt_allowance,
        "other_known_artifact_allowance_inodes": 256,
        "known_compact_peak_new_inodes_estimate": known_peak,
        "shortfall_preserving_existing_guard_and_known_peak": max(
            0, DEFAULT_MIN_FREE_INODES + known_peak - free_inodes
        ),
        "candidate_parent_pair_upper_bound": pair_rows,
        "vector_dimension": vector_dimension,
        "vector_itemsize": vector_itemsize,
        "pair_sealed_bytes_upper_bound": pair_bytes,
        "pair_consolidation_peak_bytes_upper_bound": None if pair_bytes is None else 2 * pair_bytes,
        "unknown_peaks": [
            "generation_graph_store_and_checkpoint_bytes",
            "exact_dbscan_component_arrays_and_temporaries",
            "chemistry_lineage_producer_artifacts",
            "unified_evaluation_cache_bytes_and_inodes",
        ],
        "per_candidate_files_required": False,
        "cleanup_performed": False,
        "full_storage_admission": "BLOCKED_UNMEASURED_PEAKS",
    }


def fresh_pair_command(
    route_spec: Mapping[str, Any], *, execution_repo: Path, output_root: Path,
) -> list[str]:
    """Draft the real new-universe command, without executing or authorizing it."""
    spec = validate_route_b_spec(route_spec, check_files=False)
    repo = Path(execution_repo)
    output = Path(output_root)
    generation = Path(spec["output_root"])
    if not repo.is_absolute() or repo.is_symlink():
        raise ValueError("execution repo must be absolute and physical")
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("common-recourse output must be fresh")
    if output == generation or generation in output.parents or output in generation.parents:
        raise ValueError("new common-recourse root must be separate from generation")
    # No source-pair, source-DBSCAN, historical-universe or resume arguments.
    return [
        str(spec["python"]), "-I", "-B",
        str(repo / PRODUCERS["pair_and_exact_recourse"]),
        "--config", str(repo / "configs/hpc.yaml"),
        "--set", "inference.fallback_to_heuristic=false",
        "--dataset", "mutagenicity", "--mode", "full",
        "--upstream-root", str(spec["upstream_root"]),
        "--dataset-dir", str(spec["dataset_dir"]),
        "--generation-dir", str(generation),
        "--distance-checkpoint", str(spec["distance_checkpoint"]),
        "--output-dir", str(output), "--parent-limit", "1448",
        "--device", "cpu", "--batch-size", "128",
        "--engine", "external_memory_exact_v1",
        "--external-dbscan-shortcut-mode", "sklearn_float64_exact_multi_component_v1",
        "--expected-sklearn-version", "1.7.2",
    ]


def inspect_closeout(
    *, repo_root: Path, resource_path: Path,
    decision: Mapping[str, Any] | None = None,
    vector_dimension: int | None = None, vector_itemsize: int | None = None,
) -> dict[str, Any]:
    """Read code/file-system metadata only; no GPU, SQLite, models or outputs."""
    repo = Path(repo_root).resolve(strict=True)
    inventory = {}
    for stage, relative in PRODUCERS.items():
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"producer is missing or indirect: {relative}")
        inventory[stage] = {"path": str(path), "sha256": file_sha256(path)}
    if decision is None:
        condition = "WAITING_FOR_ACTUAL_SCIENTIFIC_AB_FAILURE"
    else:
        validate_route_b_evidence(decision, check_files=True)
        condition = "SCIENTIFIC_AB_FAILURE_REOPENED"
    stat = os.statvfs(resource_path)
    resources = compact_resource_estimate(
        free_inodes=stat.f_favail, free_bytes=stat.f_bavail * stat.f_frsize,
        vector_dimension=vector_dimension, vector_itemsize=vector_itemsize,
    )
    return {
        "schema_version": SCHEMA,
        "status": "BLOCKED_SCIENCE_CRITICAL_LINEAGE_PRODUCER_MISSING",
        "dataset": "mutagenicity", "method": "comrecgc",
        "scientific_trigger_state": condition,
        "route_b_started": False, "fresh_50k_started": False,
        "pair_store_recomputed": False, "dbscan_recomputed": False,
        "matrix_written": False, "gpu_lock_acquired": False,
        "source_inventory": inventory, "resource_estimate": resources,
        "science_critical_blocker": {
            "producer": "trace-off generation at pinned A/B commit",
            "missing": "complete selected-action causal history bound to fresh candidate universe",
            "consumer": "run_mutagenicity_chemistry_audit: exact action replay and deterministic repair",
            "not_substitutable": ["old trace-on lineage", "node-origin metadata", "final graph alone"],
            "repair_requirement": "observational lineage producer with unchanged scientific state, RNG, trace-off contract and 500-step/reload parity",
        },
        "additional_wiring_required_after_lineage": [
            "fresh generation terminal and payload SHA freeze",
            "new pair manifest and exact DBSCAN transitive universe binding",
            "fresh standardized Route-B terminal validator",
            "existing canonical publisher narrow Route-B terminal dispatch",
        ],
        "historical_pair_dbscan_reuse_allowed": False,
        "launch_admission": False,
    }
