from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.utils.final_four_cells_observer import (
    FinalFourObserverError,
    REMAINING_CELLS,
    read_matrix_authority,
    snapshot,
)


def _matrix(path: Path, cells: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "fast16_matrix_authority_pointer_v1",
                "latest_count": len(cells),
                "applied_cells": cells,
                "latest_authority_root": "/runtime/final",
                "latest_combined_audit_sha256": "a" * 64,
                "latest_matrix_status_sha256": "b" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_observer_is_read_only_and_reports_the_exact_four_missing_cells(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    existing = [
        "AIDS/Ours",
        "AIDS/GCFExplainer",
        "AIDS/GlobalGCE",
        "AIDS/ComRecGC",
        "Mutagenicity/Ours",
        "Mutagenicity/GCFExplainer",
        "Mutagenicity/GlobalGCE",
        "BACE/Ours",
        "BACE/GCFExplainer",
        "BACE/GlobalGCE",
        "BACE/ComRecGC",
        "TasteMolNet/Ours",
    ]
    _matrix(path, existing)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = snapshot(matrix_authority=path)
    assert result["state"] == "RUNNING_LONG_EXPERIMENTS"
    assert result["missing_cells"] == list(REMAINING_CELLS)
    assert result["matrix_write_performed"] is False
    assert result["science_restart_performed"] is False
    assert result["llm_ablation_started"] is False
    assert result["gnn_ablation_started"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_observer_only_passes_after_all_four_cells_are_published(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    all_cells = [f"existing/{index}" for index in range(12)] + list(REMAINING_CELLS)
    _matrix(path, all_cells)
    result = snapshot(matrix_authority=path)
    assert result["state"] == "PASS"
    assert result["matrix_complete_cells"] == 16
    assert result["missing_cells"] == []


def test_matrix_authority_symlink_is_rejected(tmp_path: Path) -> None:
    physical = tmp_path / "physical.json"
    _matrix(physical, [])
    link = tmp_path / "state.json"
    link.symlink_to(physical)
    with pytest.raises(FinalFourObserverError, match="physical"):
        read_matrix_authority(link)


def test_launcher_holds_no_gpu_and_forbids_ablations() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / "scripts/autodl/launch_final_four_cells_v1.sh"
    ).read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert 'RUN_LLM_ABLATION' in text
    assert 'RUN_GNN_ABLATION' in text
    assert "nvidia-smi" not in text
