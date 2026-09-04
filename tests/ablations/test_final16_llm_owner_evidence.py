from __future__ import annotations

from pathlib import Path
import json

import pytest

from src.ablations.llm.contracts import LLMAblationContractError
from src.ablations.llm.early_launch_gate import EarlyLaunchSnapshot
from src.ablations.llm.final16_owner_evidence import (
    assert_snapshot_matches_owner_coverage,
    evaluate_final16_owner_coverage,
)
from src.ablations.launch_gate import EXPECTED_MAIN_CELL_NAMES
from src.utils.final16_owner_registry_v1 import build_owner_registry
from scripts.autodl import build_llm_early_launch_snapshot_v1 as snapshot_builder


COMMIT = "a" * 40
SPEC_SHA = "b" * 64
INCOMPLETE = (
    "TasteMolNet/GCFExplainer",
    "TasteMolNet/GlobalGCE",
    "TasteMolNet/ComRecGC",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _task(tmp_path: Path, cell: str, index: int, *, state: str = "RUNNING") -> dict:
    dataset, method = cell.split("/", 1)
    running = state == "RUNNING"
    return {
        "task_id": f"main-owner-{index}",
        "dataset": dataset,
        "method": method,
        "stage": "FINAL_MAIN_SCIENCE",
        "owner_state": state,
        "owner_pid": 1000 + index if running else None,
        "owner_start_ticks": 2000 + index if running else None,
        "heartbeat": str(tmp_path / f"owner-{index}.heartbeat.json") if running else None,
        "input_root": str(tmp_path / f"input-{index}"),
        "output_root": str(tmp_path / f"output-{index}"),
        "execution_commit": COMMIT,
        "task_spec_sha": SPEC_SHA,
        "gpu": index if running else None,
        "successor_task_id": None,
        "publisher_id": f"publisher-{index}",
    }


def _publisher(tmp_path: Path, cell: str, index: int) -> dict:
    return {
        "publisher_id": f"publisher-{index}",
        "cell_id": cell,
        "owner_state": "PREDEPLOYED",
        "owner_pid": None,
        "owner_start_ticks": None,
        "heartbeat": None,
        "locator": str(tmp_path / f"publisher-{index}.locator.json"),
        "lease_path": str(tmp_path / f"publisher-{index}.lease"),
        "execution_commit": COMMIT,
        "claim_enabled": True,
        "active_writer_count": 0,
    }


def _authority(tmp_path: Path) -> dict:
    root = tmp_path / "authority"
    root.mkdir(exist_ok=True)
    return {
        "root": str(root.resolve()),
        "complete_cells": 13,
        "applied_cells": list(EXPECTED_MAIN_CELL_NAMES[:13]),
    }


def _registry(
    tmp_path: Path,
    *,
    t14_state: str = "RUNNING",
    t8_state: str = "RUNNING",
    omit_publisher: str | None = None,
) -> dict:
    authority = _authority(tmp_path)
    tasks = [
        _task(
            tmp_path,
            cell,
            index,
            state=(
                t14_state
                if cell == "TasteMolNet/ComRecGC"
                else t8_state
                if cell == "TasteMolNet/GlobalGCE"
                else "RUNNING"
            ),
        )
        for index, cell in enumerate(INCOMPLETE)
    ]
    if omit_publisher is not None:
        for task, cell in zip(tasks, INCOMPLETE, strict=True):
            if cell == omit_publisher:
                task["publisher_id"] = None
    publishers = [
        _publisher(tmp_path, cell, index)
        for index, cell in enumerate(INCOMPLETE)
        if cell != omit_publisher
    ]
    return build_owner_registry(
        registry_id="final16-owner-fixture",
        matrix_authority_root=authority["root"],
        tasks=tasks,
        publishers=publishers,
        gpu_leases=[],
        check_processes=False,
    )


def _snapshot(
    tmp_path: Path,
    coverage,
    *,
    t8_pid: int | None = 1001,
    t8_state: str = "RUNNING",
) -> EarlyLaunchSnapshot:
    return EarlyLaunchSnapshot(
        matrix_complete_cells=len(coverage.applied_cells),
        matrix_authority_path=str((tmp_path / "matrix-state.json").resolve()),
        matrix_authority_sha256="c" * 64,
        t8_t13_state=t8_state,
        t8_t13_science_pid=t8_pid,
        t12_healthy=True,
        t14_healthy=True,
        mut_passed_or_gpu_released=True,
        main_ready_waiting_gpu=(),
        main_publishers_waiting_gpu=(),
        idle_gpu=3,
        idle_gpu_seconds=1200,
        persistent_free_gb=500.0,
        minimum_persistent_free_gb=100.0,
        memory_available_gb=128.0,
        minimum_memory_available_gb=64.0,
        checkpoint_resume_supported=True,
        requested_early_gpus=1,
        main_owner_registry_path=str((tmp_path / "owner-registry.json").resolve()),
        main_owner_registry_sha256="d" * 64,
        main_owner_registry_self_sha256=coverage.registry_self_sha256,
        all_incomplete_main_cells_owned=coverage.all_incomplete_cells_owned,
        unhealthy_or_unowned_main_cells=coverage.unhealthy_or_unowned_cells,
        missing_main_publisher_cells=coverage.missing_publisher_cells,
        active_early_llm_ablation_gpus=(),
    )


def test_all_three_unfinished_taste_cells_require_live_owner_and_publisher(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=_registry(tmp_path),
        check_processes=False,
    )
    assert coverage.incomplete_cells == INCOMPLETE
    assert coverage.healthy_owner_cells == INCOMPLETE
    assert coverage.unhealthy_or_unowned_cells == ()
    assert coverage.missing_publisher_cells == ()
    assert coverage.all_incomplete_cells_owned is True
    assert_snapshot_matches_owner_coverage(_snapshot(tmp_path, coverage), coverage)


def test_registry_may_bind_the_unique_pointer_root(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    pointer_root = tmp_path / "fast16-matrix-authority"
    pointer_root.mkdir()
    authority["pointer_root"] = str(pointer_root.resolve())
    tasks = [_task(tmp_path, cell, index) for index, cell in enumerate(INCOMPLETE)]
    publishers = [
        _publisher(tmp_path, cell, index) for index, cell in enumerate(INCOMPLETE)
    ]
    registry = build_owner_registry(
        registry_id="pointer-root-fixture",
        matrix_authority_root=pointer_root,
        tasks=tasks,
        publishers=publishers,
        gpu_leases=[],
        check_processes=False,
    )
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=registry,
        check_processes=False,
    )
    assert coverage.matrix_authority_root == str(pointer_root.resolve())


def test_predeployed_owner_and_missing_publisher_do_not_open_llm_gate(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=_registry(
            tmp_path,
            t14_state="PREDEPLOYED",
            omit_publisher="TasteMolNet/GCFExplainer",
        ),
        check_processes=False,
    )
    assert coverage.unhealthy_or_unowned_cells == ("TasteMolNet/ComRecGC",)
    assert coverage.missing_publisher_cells == ("TasteMolNet/GCFExplainer",)
    assert coverage.all_incomplete_cells_owned is False


def test_snapshot_cannot_claim_another_running_t8_owner(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=_registry(tmp_path),
        check_processes=False,
    )
    with pytest.raises(LLMAblationContractError, match="T8/T13 PID"):
        assert_snapshot_matches_owner_coverage(
            _snapshot(tmp_path, coverage, t8_pid=9999), coverage
        )


def test_sealed_t8_pass_owner_is_healthy_while_publisher_appends(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=_registry(tmp_path, t8_state="PASS"),
        check_processes=False,
    )
    snapshot = _snapshot(tmp_path, coverage, t8_pid=None, t8_state="PASS")
    assert_snapshot_matches_owner_coverage(snapshot, coverage)


def test_mut_gpu_release_is_not_a_substitute_for_registered_mut_pass(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    authority["complete_cells"] = 12
    authority["applied_cells"] = list(EXPECTED_MAIN_CELL_NAMES[:7]) + list(
        EXPECTED_MAIN_CELL_NAMES[8:13]
    )
    registry = _registry(tmp_path)
    coverage = evaluate_final16_owner_coverage(
        authority=authority,
        owner_registry=registry,
        check_processes=False,
    )
    snapshot = _snapshot(tmp_path, coverage)
    with pytest.raises(LLMAblationContractError, match="registered Mut PASS"):
        assert_snapshot_matches_owner_coverage(snapshot, coverage)


def test_snapshot_builder_selects_only_truly_idle_gpu_and_binds_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    matrix = tmp_path / "matrix-state.json"
    matrix.write_text("{}\n", encoding="utf-8")
    registry = _registry(tmp_path)
    registry_path = tmp_path / "owner-registry.json"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        snapshot_builder,
        "validate_matrix_authority_pointer",
        lambda _payload: authority,
    )
    observation = {
        "schema_version": snapshot_builder.OBSERVATION_SCHEMA,
        "main_ready_waiting_gpu": [],
        "main_publishers_waiting_gpu": [],
        "gpus": [
            {
                "index": 0,
                "idle_seconds": 9999,
                "compute_pids": [99],
                "main_lease_held": False,
                "ablation_family": None,
            },
            {
                "index": 1,
                "idle_seconds": 1300,
                "compute_pids": [],
                "main_lease_held": False,
                "ablation_family": None,
            },
            {
                "index": 2,
                "idle_seconds": 5000,
                "compute_pids": [],
                "main_lease_held": True,
                "ablation_family": None,
            },
            {
                "index": 3,
                "idle_seconds": 0,
                "compute_pids": [101],
                "main_lease_held": False,
                "ablation_family": "llm",
            },
        ],
        "persistent_free_gb": 500.0,
        "minimum_persistent_free_gb": 100.0,
        "memory_available_gb": 128.0,
        "minimum_memory_available_gb": 64.0,
        "checkpoint_resume_supported": True,
    }
    snapshot = snapshot_builder.build_snapshot(
        matrix_path=matrix,
        owner_registry_path=registry_path,
        runtime_observation=observation,
        check_processes=False,
    )
    assert snapshot.idle_gpu == 1
    assert snapshot.idle_gpu_seconds == 1300
    assert snapshot.active_early_llm_ablation_gpus == (3,)
    assert snapshot.main_owner_registry_self_sha256 == registry["self_sha256"]


def test_snapshot_builder_has_thin_paired_slurm_entrypoint() -> None:
    python_entrypoint = (
        PROJECT_ROOT / "scripts/autodl/build_llm_early_launch_snapshot_v1.py"
    )
    slurm = PROJECT_ROOT / "scripts/slurm/build_llm_early_launch_snapshot_v1.sh"
    assert python_entrypoint.is_file() and slurm.is_file()
    source = slurm.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert required in source
