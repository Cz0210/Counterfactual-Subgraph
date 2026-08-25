from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from scripts.autodl import run_aids_comrecgc_exact_recovery_stage as stage_cli
from src.baselines.comrecgc.failed_selection_adoption import PRODUCTION_AUTHORITY
from src.utils import autodl_aids_comrecgc_exact_recovery_controller_v1 as controller
from src.utils import autodl_aids_comrecgc_exact_recovery_spec_v1 as builder


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file(path: Path, value: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "path": str(path),
        "roles": [role],
        "sha256": controller.sha256_file(path),
    }


def test_controller_projection_contract_matches_reviewed_adoption_profile() -> None:
    assert controller.EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256 == {
        "close": PRODUCTION_AUTHORITY.close_state_authority.projection_sha256,
        "final": PRODUCTION_AUTHORITY.final_state_authority.projection_sha256,
    }


def test_production_spec_is_derived_from_typed_receipt_and_builds_native_dag(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source_manifest = _json(source / "controller.json", {"source": True})
    close_gate = _json(source / "close-gate.json", {"status": "PASS"})
    final_gate = _json(source / "final-gate.json", {"status": "FAILED"})
    close_manifest = _json(source / "close.json", {"status": "PASS"})
    pair_semantics = _json(source / "pair-semantics.json", {"status": "PASS"})
    pair_store = _json(source / "pair-store.json", {"run_complete": True})
    pairs = _file(source / "pairs.npy")
    vectors = _file(source / "vectors.npy")
    distances = _file(source / "distances.npy")
    bitmap = _file(source / "bitmap.npy")
    checkpoint = _json(source / "checkpoint.json", {"phase": "shortcut_blocked"})
    selection = _json(source / "selection.json", {"status": "PASS"})
    failure = _json(source / "failure.json", {"status": "INCONCLUSIVE"})
    failure_indices = _file(source / "failure-indices.npy")
    anchor_indices = _file(source / "anchor-indices.npy")
    anchor_rows = _file(source / "anchor-rows.npy")
    anchor_edges = _file(source / "anchor-edges.npy")
    role_rows = [
        _artifact(anchor_indices, "adaptive selected anchor indices"),
        _artifact(anchor_rows, "adaptive selected anchor rows"),
        _artifact(failure_indices, "adaptive first-pass failure indices"),
        _artifact(anchor_edges, "failed disconnected anchor edges"),
        _artifact(pair_store, "physical pair-store manifest"),
        _artifact(distances, "normalized distance authority"),
        _artifact(bitmap, "close bitmap"),
    ]
    runtime_dirs = {
        key: tmp_path / "runtime" / key
        for key in (
            "SOURCE_GENERATION_ROOT",
            "COMRECGC_UPSTREAM_ROOT",
            "DATASET_DIR",
            "MOLCLR_ROOT",
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT",
        )
    }
    for path in runtime_dirs.values():
        path.mkdir(parents=True)
    runtime_files = {
        key: _file(tmp_path / "runtime" / f"{key}.bin")
        for key in (
            "SOURCE_CSV",
            "DISTANCE_CHECKPOINT",
            "DATASET_CSV",
            "TEACHER_PATH",
            "MOLCLR_CHECKPOINT",
            "THRESHOLDS_PATH",
        )
    }
    environment = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
        "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
        **{key: str(value) for key, value in runtime_dirs.items()},
        **{key: str(value) for key, value in runtime_files.items()},
    }
    receipt = {
        "schema_version": "aids_comrecgc_c766_failed_selection_adoption_v3",
        "status": "RECOVERY_ONLY_READY",
        "artifact_kind": "aids_c766_failed_selection_recovery_evidence_v3",
        "terminal_marker": "RECOVERY_EVIDENCE_READY",
        "authority_profile_sha256": "a" * 64,
        "authority": {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": controller.sha256_file(source_manifest),
            "close_gate": str(close_gate),
            "close_gate_sha256": controller.sha256_file(close_gate),
            "final_gate": str(final_gate),
            "final_gate_sha256": controller.sha256_file(final_gate),
        },
        "close_authority": {
            "manifest": str(close_manifest),
            "manifest_sha256": controller.sha256_file(close_manifest),
            "all_pairs_close": True,
            "physical_rows": controller.EXPECTED_ROWS,
            "logical_close_rows": controller.EXPECTED_ROWS,
            "pair_semantics_contract": str(pair_semantics),
            "pair_semantics_contract_sha256": controller.sha256_file(
                pair_semantics
            ),
            "pair_path": str(pairs),
            "pair_sha256": controller.sha256_file(pairs),
            "vector_path": str(vectors),
            "vector_sha256": controller.sha256_file(vectors),
        },
        "failed_selection": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": controller.sha256_file(checkpoint),
            "selection_manifest": str(selection),
            "selection_manifest_sha256": controller.sha256_file(selection),
            "failure_artifact": str(failure),
            "failure_artifact_sha256": controller.sha256_file(failure),
            "anchor_count": 266,
            "unique_seed_component": True,
            "dbscan_partition_proven": False,
        },
        "final_task": {"task_id": "final", "expected_output": str(source)},
        "task_state_authority": {
            "close_projection_sha256": controller.EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256[
                "close"
            ],
            "final_projection_sha256": controller.EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256[
                "final"
            ],
        },
        "source_artifacts": role_rows,
    }
    adoption_parent = tmp_path / "adoption-parent"
    adoption_output = adoption_parent / "receipt-child"
    adoption_output.mkdir(parents=True)
    receipt_path = _json(
        adoption_output / "failed_selection_adoption_receipt.json", receipt
    )
    _file(adoption_output / "RECOVERY_EVIDENCE_READY", b"READY\n")

    module_file = _file(tmp_path / "failed_selection_adoption.py")
    fake_module = ModuleType(builder.ADOPTION_MODULE)
    fake_module.__file__ = str(module_file)
    fake_module.verify_aids_c766_failed_selection_recovery_evidence = (
        lambda *, output_dir: receipt
    )
    monkeypatch.setitem(sys.modules, builder.ADOPTION_MODULE, fake_module)
    monkeypatch.setattr(
        builder,
        "ADOPTION_ENTRYPOINT",
        "scripts/autodl/status_aids_comrecgc_exact_recovery.py",
    )
    fake_controller = SimpleNamespace(
        by_id={"final": SimpleNamespace(environment=environment)}
    )
    project_root = Path(__file__).resolve().parents[2]
    controller_parent = tmp_path / "controllers"
    controller_parent.mkdir()
    manifests = controller_parent / "manifests"
    manifests.mkdir()
    controller_manifest = manifests / "controller.json"
    spec = builder.generate_production_spec(
        adoption_output=adoption_output,
        controller_parent=controller_parent,
        python=Path(sys.executable).resolve(strict=True),
        project_root=project_root,
        controller_manifest_path=controller_manifest,
        timestamp="20260825T010203Z",
        adoption_validator=lambda *, output_dir: receipt,
        manifest_loader=lambda path: fake_controller,
    )
    spec_path = _json(tmp_path / "generated-spec.json", spec)
    built = controller.build_controller_manifest(
        spec_path=spec_path, output_path=controller_manifest
    )
    expected_pins = {
        "science_commit": controller.SCIENCE_RELEASE_COMMIT,
        "adoption_commit": builder.ADOPTION_RELEASE_COMMIT,
        "controller_commit": builder.CONTROLLER_RELEASE_COMMIT,
        "exact_runner_commit": builder.EXACT_RUNNER_RELEASE_COMMIT,
        "subset_runner_commit": builder.SUBSET_RUNNER_RELEASE_COMMIT,
        "downstream_runner_commit": builder.DOWNSTREAM_RUNNER_RELEASE_COMMIT,
        "standardization_runner_commit": (
            builder.STANDARDIZATION_RUNNER_RELEASE_COMMIT
        ),
    }
    assert built["release_pins"] == expected_pins
    assert built["release_ready"] is all(
        isinstance(value, str) and len(value) == 40
        for value in expected_pins.values()
    )
    assert built["production_deployment_authorized"] is (
        builder.PRODUCTION_DEPLOYMENT_AUTHORIZED
    )
    stages = {row["stage_id"]: row for row in built["stages"]}
    assert stages[controller.EXACT_STAGE]["output_dir"].endswith(
        "/science/common_recourse/external_memory"
    )
    assert stages[controller.DOWNSTREAM_STAGE]["output_dir"].endswith(
        "/science/common_recourse/external_memory/all_core_component_summary"
    )
    assert stages[controller.FINAL_STAGE]["output_dir"].endswith("/science")
    for stage_id in (
        controller.SUBSET_STAGE,
        controller.EXACT_STAGE,
        controller.DOWNSTREAM_STAGE,
        controller.FINAL_STAGE,
    ):
        argv = stages[stage_id]["commands"]["fresh"]
        parsed = stage_cli.build_parser().parse_args(argv[2:])
        assert parsed.controller_manifest == controller_manifest
    assert controller.sha256_file(receipt_path) == controller.sha256_file(
        stages[controller.ADOPTION_STAGE]["terminal_path"]
    )
