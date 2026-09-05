from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.utils import autodl_mut_route_b_closeout_v1 as preflight


ROOT = Path(__file__).resolve().parents[2]


def test_compact_files_do_not_scale_per_candidate() -> None:
    result = preflight.compact_resource_estimate(free_inodes=98_232, free_bytes=10**12)
    assert result["pair_chunk_count_upper_bound"] == 782
    assert result["pair_chunk_inodes"] == 1564
    assert result["known_compact_peak_new_inodes_estimate"] == 2068
    assert result["existing_guard_shortfall_inodes"] == 1768
    assert result["shortfall_preserving_existing_guard_and_known_peak"] == 3836
    assert result["guard_modified"] is False
    assert result["full_storage_admission"] == "BLOCKED_UNMEASURED_PEAKS"


def test_resource_estimate_keeps_unknown_bytes_unknown() -> None:
    result = preflight.compact_resource_estimate(free_inodes=200_000, free_bytes=10**13)
    assert result["pair_sealed_bytes_upper_bound"] is None
    assert result["pair_consolidation_peak_bytes_upper_bound"] is None
    assert result["unknown_peaks"]
    assert result["status"] != "PASS"


def test_pair_storage_includes_consolidation_and_new_universe_upper_bound() -> None:
    result = preflight.compact_resource_estimate(
        free_inodes=200_000, free_bytes=10**13, vector_dimension=64, vector_itemsize=4,
    )
    assert result["candidate_parent_pair_upper_bound"] == 144_800_000
    assert result["pair_sealed_bytes_upper_bound"] == 144_800_000 * 272
    assert result["pair_consolidation_peak_bytes_upper_bound"] == 2 * 144_800_000 * 272


@pytest.mark.parametrize("kwargs", [
    {"free_inodes": -1, "free_bytes": 1},
    {"free_inodes": True, "free_bytes": 1},
    {"free_inodes": 1, "free_bytes": 1, "vector_dimension": 64},
    {"free_inodes": 1, "free_bytes": 1, "vector_dimension": 64, "vector_itemsize": 2},
])
def test_invalid_resource_claim_fails(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        preflight.compact_resource_estimate(**kwargs)


def test_no_decision_never_authorizes_generation(tmp_path: Path) -> None:
    result = preflight.inspect_closeout(repo_root=ROOT, resource_path=tmp_path)
    assert result["status"] == "BLOCKED_SCIENCE_CRITICAL_LINEAGE_PRODUCER_MISSING"
    assert result["scientific_trigger_state"] == "WAITING_FOR_ACTUAL_SCIENTIFIC_AB_FAILURE"
    for field in ("route_b_started", "fresh_50k_started", "pair_store_recomputed",
                  "dbscan_recomputed", "matrix_written", "gpu_lock_acquired", "launch_admission"):
        assert result[field] is False
    assert all(len(row["sha256"]) == 64 for row in result["source_inventory"].values())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("decision", [
    {"classification": "ENGINEERING_FAILURE", "status": "FAILED"},
    {"classification": "SCIENTIFIC_STATE_DIVERGENCE", "status": "READY"},
    {"classification": "EXACT_SEMANTIC_EQUIVALENCE", "status": "PASS"},
])
def test_unsealed_or_non_scientific_failure_cannot_select_route_b(
    tmp_path: Path, decision: dict,
) -> None:
    with pytest.raises(Exception):
        preflight.inspect_closeout(repo_root=ROOT, resource_path=tmp_path, decision=decision)
    assert not list(tmp_path.iterdir())


def test_draft_pair_command_never_reads_old_universe(monkeypatch, tmp_path: Path) -> None:
    spec = {
        "python": "/env/python", "upstream_root": "/upstream",
        "dataset_dir": "/dataset", "distance_checkpoint": "/distance.pt",
        "output_root": str(tmp_path / "new-generation"),
    }
    seen = []

    def validate(value, *, check_files):
        seen.append(check_files)
        return value

    monkeypatch.setattr(preflight, "validate_route_b_spec", validate)
    command = preflight.fresh_pair_command(spec, execution_repo=ROOT, output_root=tmp_path / "new-common")
    assert seen == [False]
    assert command[command.index("--generation-dir") + 1] == spec["output_root"]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--engine") + 1] == "external_memory_exact_v1"
    assert not any("source-manifest" in arg or "source-checkpoint" in arg for arg in command)
    assert "--resume" not in command
    with pytest.raises(ValueError, match="separate"):
        preflight.fresh_pair_command(spec, execution_repo=ROOT, output_root=tmp_path / "new-generation" / "common")


def test_real_pair_store_rejects_changed_generation_universe(tmp_path: Path) -> None:
    from src.baselines.comrecgc.external_memory_recourse import ExternalPairStore, ExternalMemoryDBSCANError

    identity = {"counterfactuals_sha256": "a" * 64, "candidate_count": 2}
    store = ExternalPairStore(root=tmp_path / "pair", scientific_identity=identity, max_rss_bytes=2**50)
    store.append(chunk_index=0, pairs=np.array([[0, 0]], dtype=np.int64),
                 vectors=np.zeros((1, 2), dtype=np.float32), chunk_identity={"candidate_start": 0})
    store.finalize()
    with pytest.raises(ExternalMemoryDBSCANError, match="identity mismatch"):
        ExternalPairStore(root=tmp_path / "pair", scientific_identity={**identity, "counterfactuals_sha256": "b" * 64},
                          max_rss_bytes=2**50, resume=True)


def test_cli_is_read_only_and_paired_slurm_cpu_only(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/autodl/preflight_mut_route_b_closeout_v1.py"),
        "--config", "configs/hpc.yaml", "--resource-path", str(tmp_path),
    ], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["launch_admission"] is False
    assert list(tmp_path.iterdir()) == before
    slurm = (ROOT / "scripts/slurm/preflight_mut_route_b_closeout_v1.sh").read_text()
    assert "#SBATCH --partition=intel" in slurm and "#SBATCH --gres" not in slurm
    assert "export PYTHONPATH=$PWD" in slurm
    assert "--config configs/hpc.yaml" in slurm
