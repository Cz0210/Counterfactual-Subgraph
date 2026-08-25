from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.autodl import run_comrecgc_standardized_continuation as continuation
from src.baselines.comrecgc.external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalDBSCANContract,
)
from src.utils import autodl_aids_comrecgc_exact_recovery_controller_v1 as controller
from src.utils import autodl_aids_comrecgc_exact_recovery_stages_v1 as stages


@pytest.fixture(autouse=True)
def _frozen_recovery_cpu_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
    }
    for field, value in expected.items():
        monkeypatch.setenv(field, value)
    monkeypatch.setattr(stages.os, "getpid", lambda: 4242)
    monkeypatch.setattr(stages.os, "getpgrp", lambda: 4242)


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file(path: Path, value: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


@pytest.mark.parametrize("window", ("temp_only", "final_and_temp"))
def test_stage_immutable_json_publication_crash_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    target = tmp_path / "terminal.json"
    temporary = tmp_path / ".terminal.json.publish.tmp"
    payload = {"schema_version": "fixture", "status": "PASS"}
    if window == "temp_only":
        real_link = stages.os.link

        def crash_before_link(source: object, destination: object, **kwargs: object) -> None:
            if Path(str(destination)) == target:
                raise RuntimeError("crash-before-stage-link")
            real_link(source, destination, **kwargs)

        monkeypatch.setattr(stages.os, "link", crash_before_link)
        with pytest.raises(RuntimeError, match="crash-before-stage-link"):
            stages._write_new_json(target, payload)
        assert temporary.is_file()
        assert not target.exists()
        monkeypatch.setattr(stages.os, "link", real_link)
        stages._write_new_json(target, payload)
    else:
        stages._write_new_json(target, payload)
        stages.os.link(target, temporary, follow_symlinks=False)
        assert stages._reconcile_immutable_stage_publication(target) is True
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert target.stat().st_nlink == 1
    assert not temporary.exists()


def test_promoted_evidence_copy_reconciles_both_link_crash_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _file(tmp_path / "source.bin", b"authenticated-evidence\n")
    target = tmp_path / "promoted/evidence.bin"
    digest = controller.sha256_file(source)
    real_link = stages.os.link

    def crash_before_link(source_path: object, target_path: object, **kwargs: object) -> None:
        if Path(str(target_path)) == target:
            raise RuntimeError("crash-before-evidence-link")
        real_link(source_path, target_path, **kwargs)

    monkeypatch.setattr(stages.os, "link", crash_before_link)
    with pytest.raises(RuntimeError, match="crash-before-evidence-link"):
        stages._copy_new_file(source, target, expected_sha256=digest)
    temporary = target.parent / ".evidence.bin.copy.tmp"
    assert temporary.is_file()
    monkeypatch.setattr(stages.os, "link", real_link)
    stages._copy_new_file(source, target, expected_sha256=digest)
    assert target.read_bytes() == source.read_bytes()
    real_link(target, temporary, follow_symlinks=False)
    stages._copy_new_file(source, target, expected_sha256=digest)
    assert target.stat().st_nlink == 1
    assert not temporary.exists()


def _npy(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return path


def _stage_row(
    stage_id: str,
    output: Path,
    terminal: Path,
    bindings: dict[str, Path],
) -> dict[str, object]:
    return {
        "stage_id": stage_id,
        "output_dir": str(output),
        "terminal_path": str(terminal),
        "argv_bindings": {
            role: {"value": str(value)} for role, value in bindings.items()
        },
    }


def test_stage_environment_rejects_gpu_or_thread_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {"resources": {"thread_count": 16}}
    assert stages._require_cpu_stage_environment(manifest)["OMP_NUM_THREADS"] == "16"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(stages.RecoveryStageError, match="CPU-only stage environment"):
        stages._require_cpu_stage_environment(manifest)


def test_final_stage_requires_controller_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages.os, "getpid", lambda: 4242)
    monkeypatch.setattr(stages.os, "getpgrp", lambda: 4000)
    with pytest.raises(stages.RecoveryStageError, match="start_new_session"):
        stages._require_controller_process_group()


def test_subset_stage_keeps_partial_attempt_and_restarts_in_fresh_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_root = tmp_path / "controller"
    output = controller_root / "subset"
    manifest_path = tmp_path / "controller.json"
    adoption_gate = controller_root / "gates/01_failed_selection_adoption.json"
    manifest = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": "a" * 64,
        "controller_root": str(controller_root),
        "stages": [
            _stage_row(
                controller.SUBSET_STAGE,
                output,
                output / "subset_stage_receipt.json",
                {
                    "controller_manifest": manifest_path,
                    "adoption_gate": adoption_gate,
                },
            )
        ],
        "source_authority": {
            "close_pair_manifest_path": str(tmp_path / "close.json"),
            "close_pair_manifest_sha256": "b" * 64,
            "physical_pairs_path": str(tmp_path / "pairs.npy"),
            "physical_pairs_sha256": "c" * 64,
        },
        "runtime_inputs": {"expected_sklearn_version": "1.7.2"},
        "resources": {"subset_size": 3, "block_size": 4, "thread_count": 16},
    }
    monkeypatch.setattr(stages, "load_bound_controller_manifest", lambda path: manifest)
    monkeypatch.setattr(
        stages,
        "open_typed_recovery_gate",
        lambda *args: {"gate_sha256": "d" * 64},
    )
    _json(adoption_gate, {"gate_sha256": "d" * 64})
    calls = 0

    def fake_audit(*, output_dir: Path, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        root = Path(output_dir)
        root.mkdir(parents=True)
        if calls == 1:
            _file(root / "partial.bin")
            raise RuntimeError("fixture interruption")
        result = {"status": "PASS"}
        _json(root / "production_subset_equivalence.json", result)
        _file(root / "PASS", b"PASS\n")
        return result

    monkeypatch.setattr(
        stages, "run_production_subset_equivalence_audit", fake_audit
    )
    terminal_reopens: list[str] = []
    monkeypatch.setattr(
        stages,
        "validate_stage_terminal",
        lambda manifest, *, stage_id: terminal_reopens.append(stage_id),
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        stages.run_subset_stage(
            controller_manifest=manifest_path,
            adoption_gate=adoption_gate,
            output_dir=output,
            resume=False,
        )
    receipt = stages.run_subset_stage(
        controller_manifest=manifest_path,
        adoption_gate=adoption_gate,
        output_dir=output,
        resume=True,
    )
    assert receipt["attempt"] == 1
    assert (output / "attempt-0/partial.bin").is_file()
    assert (output / "attempt-1/PASS").read_bytes() == b"PASS\n"
    assert stages.run_subset_stage(
        controller_manifest=manifest_path,
        adoption_gate=adoption_gate,
        output_dir=output,
        resume=True,
    ) == receipt
    assert terminal_reopens == [controller.SUBSET_STAGE]


def test_exact_stage_archives_13_small_files_bootstraps_and_never_copies_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_root = tmp_path / "controller"
    science = controller_root / "science"
    output = science / "common_recourse/external_memory"
    terminal = output / "exact_recovery_receipt.json"
    manifest_path = tmp_path / "controller.json"
    adoption_gate = controller_root / "gates/01_failed_selection_adoption.json"
    subset_gate = controller_root / "gates/02_production_subset_equivalence.json"
    failed_root = tmp_path / "failed-source"
    inventory: list[dict[str, str]] = []
    names = ["FAILED.json", *[f"evidence/{index}.bin" for index in range(13)]]
    for index, name in enumerate(names):
        path = _file(failed_root / name, f"row-{index}\n".encode())
        inventory.append(
            {
                "relative_path": name,
                "sha256": controller.sha256_file(path),
            }
        )
    receipt_path = tmp_path / "adoption/failed_selection_adoption_receipt.json"
    receipt = {
        "failed_tree_inventory": inventory,
        "final_task": {"expected_output": str(failed_root)},
    }
    _json(receipt_path, receipt)
    vectors = _file(tmp_path / "vectors.npy")
    checkpoint_path = _json(tmp_path / "checkpoint.json", {"source": True})
    selection_path = _json(tmp_path / "selection.json", {"source": True})
    failure_path = _json(tmp_path / "failure.json", {"source": True})
    contract = ExternalDBSCANContract(
        eps=0.02,
        min_samples=3,
        query_block_size=4,
        checkpoint_interval_blocks=1,
        max_rss_bytes=controller.DEFAULT_MAX_RSS_BYTES,
        expected_sklearn_version="1.7.2",
        shortcut_mode=ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    )
    source = {
        "failed_checkpoint_path": str(checkpoint_path),
        "failed_checkpoint_sha256": controller.sha256_file(checkpoint_path),
        "adaptive_selection_path": str(selection_path),
        "adaptive_selection_sha256": controller.sha256_file(selection_path),
        "failed_shortcut_artifact_path": str(failure_path),
        "failed_shortcut_artifact_sha256": controller.sha256_file(failure_path),
        "failure_indices_sha256": "1" * 64,
        "anchor_indices_sha256": "2" * 64,
        "anchor_rows_sha256": "3" * 64,
        "source_vectors_path": str(vectors),
        "source_vectors_sha256": controller.sha256_file(vectors),
        "pair_store_manifest_path": str(tmp_path / "pair-store-manifest.json"),
        "close_pair_manifest_path": str(tmp_path / "close-pair-manifest.json"),
    }
    runtime = {
        key: str(tmp_path / key)
        for key in (
            "source_generation_root",
            "upstream_root",
            "dataset_dir",
            "source_csv",
            "distance_checkpoint",
            "dataset_csv",
            "teacher_path",
            "molclr_root",
            "molclr_checkpoint",
            "thresholds_path",
            "pair_store_owner_root",
        )
    }
    runtime.update(
        {"expected_sklearn_version": "1.7.2", "theta_star": 0.05, "cost_cap": 0.0535}
    )
    manifest = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": "a" * 64,
        "controller_root": str(controller_root),
        "adoption_contract": {
            "expected_task_state_projection_sha256": {"close": "b" * 64, "final": "c" * 64}
        },
        "source_authority": source,
        "runtime_inputs": runtime,
        "resources": {
            "max_rss_bytes": controller.DEFAULT_MAX_RSS_BYTES,
            "max_rss_scope": (
                "exact_dbscan_process_with_native_peak_certificate"
            ),
            "block_size": 4,
            "thread_count": 16,
        },
        "stages": [
            _stage_row(
                controller.EXACT_STAGE,
                output,
                terminal,
                {
                    "controller_manifest": manifest_path,
                    "adoption_gate": adoption_gate,
                    "subset_gate": subset_gate,
                },
            ),
            _stage_row(controller.FINAL_STAGE, science, science / "final.json", {}),
        ],
    }
    monkeypatch.setattr(stages, "load_bound_controller_manifest", lambda path: manifest)
    monkeypatch.setattr(stages, "open_typed_recovery_gate", lambda *args: {})
    monkeypatch.setattr(
        stages,
        "validate_typed_adoption_receipt",
        lambda **kwargs: {
            "receipt": receipt,
            "receipt_path": str(receipt_path),
            "receipt_sha256": controller.sha256_file(receipt_path),
        },
    )
    monkeypatch.setattr(
        stages,
        "_load_checkpoint",
        lambda path: {"identity": {"contract": asdict(contract)}},
    )

    def fake_bootstrap(inputs: object) -> dict[str, object]:
        science.mkdir(parents=True, exist_ok=True)
        value = {
            "status": "READY_FOR_EXTERNAL_COMMON_RECOVERY",
            "output_root": str(science),
            "common_recourse_started": False,
            "downstream_started": False,
        }
        _json(science / "exact_recovery_continuation_bootstrap.json", value)
        return value

    monkeypatch.setattr(
        continuation,
        "bootstrap_external_common_recovery_continuation",
        fake_bootstrap,
    )
    promotion_manifest = tmp_path / "promotion.json"
    _json(promotion_manifest, {"status": "PASS"})
    monkeypatch.setattr(
        stages,
        "promote_failed_adaptive_selection_for_component_recovery",
        lambda **kwargs: SimpleNamespace(
            promotion_manifest_path=promotion_manifest,
            promotion_manifest_sha256=controller.sha256_file(promotion_manifest),
        ),
    )
    dbscan_manifest = output / "dbscan/run_manifest.json"

    def fake_fit(**kwargs: object) -> SimpleNamespace:
        _json(
            dbscan_manifest,
            {
                "clustering_path": ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
                "approximation_used": False,
            },
        )
        return SimpleNamespace(
            manifest_path=dbscan_manifest,
            manifest_sha256=controller.sha256_file(dbscan_manifest),
        )

    monkeypatch.setattr(
        stages,
        "fit_promoted_failed_selection_component_recovery",
        fake_fit,
    )
    monkeypatch.setattr(stages, "_validate_component_recovery_closure", lambda **kwargs: None)
    result = stages.run_exact_stage(
        controller_manifest=manifest_path,
        adoption_gate=adoption_gate,
        subset_gate=subset_gate,
        output_dir=output,
        resume=False,
    )
    assert result["promoted_source_artifact_count"] == 13
    assert result["recovery_source_authority"]["seed_failure_scan_reexecuted"] is False
    assert not (output / "source_evidence/FAILED.json").exists()
    assert len(list((output / "source_evidence/evidence").glob("*.bin"))) == 13
    assert vectors.read_bytes() == b"fixture\n"
    assert stages.run_exact_stage(
        controller_manifest=manifest_path,
        adoption_gate=adoption_gate,
        subset_gate=subset_gate,
        output_dir=output,
        resume=True,
    ) == result


def test_partial_final_owner_requires_explicit_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "science"
    owner = output / ".exact-recovery-final-owner"
    owner.mkdir(parents=True)
    manifest_path = tmp_path / "controller.json"
    bindings = {
        "controller_manifest": manifest_path,
        "adoption_gate": tmp_path / "adoption-gate.json",
        "subset_gate": tmp_path / "subset-gate.json",
        "exact_gate": tmp_path / "exact-gate.json",
        "downstream_gate": tmp_path / "downstream-gate.json",
    }
    manifest = {
        "resources": {"thread_count": 16},
        "stages": [
            _stage_row(
                controller.FINAL_STAGE,
                output,
                output / "exact_recovery_freeze_receipt.json",
                bindings,
            )
        ]
    }
    monkeypatch.setattr(stages, "load_bound_controller_manifest", lambda path: manifest)
    with pytest.raises(
        stages.RecoveryStageError,
        match="requires explicit resume authorization",
    ):
        stages.run_final_stage(
            controller_manifest=manifest_path,
            adoption_gate=bindings["adoption_gate"],
            subset_gate=bindings["subset_gate"],
            exact_gate=bindings["exact_gate"],
            downstream_gate=bindings["downstream_gate"],
            output_dir=output,
            resume=False,
        )


def test_downstream_stage_uses_native_component_summary_and_full_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_root = tmp_path / "controller"
    exact_root = controller_root / "science/common_recourse/external_memory"
    output = exact_root / "all_core_component_summary"
    manifest_path = tmp_path / "controller.json"
    exact_gate = controller_root / "gates/03_exact_component_recovery.json"
    _json(exact_gate, {"gate_sha256": "e" * 64})
    labels_path = _npy(exact_root / "dbscan/labels.npy", np.zeros(3, dtype=np.int64))
    dbscan_path = _json(
        exact_root / "dbscan/run_manifest.json",
        {"labels_path": str(labels_path)},
    )
    exact_receipt = _json(
        exact_root / "exact_recovery_receipt.json",
        {
            "dbscan_manifest_path": str(dbscan_path),
            "dbscan_manifest_sha256": controller.sha256_file(dbscan_path),
        },
    )
    manifest = {
        "controller_root": str(controller_root),
        "stages": [
            _stage_row(
                controller.DOWNSTREAM_STAGE,
                output,
                output / "run_manifest.json",
                {
                    "controller_manifest": manifest_path,
                    "exact_gate": exact_gate,
                },
            )
        ],
        "source_authority": {"close_pair_manifest_path": str(tmp_path / "close.json")},
        "runtime_inputs": {"upstream_root": str(tmp_path / "upstream")},
        "resources": {
            "max_rss_bytes": controller.DEFAULT_MAX_RSS_BYTES,
            "max_rss_scope": (
                "exact_dbscan_process_with_native_peak_certificate"
            ),
            "block_size": 4,
            "thread_count": 16,
        },
    }
    monkeypatch.setattr(stages, "load_bound_controller_manifest", lambda path: manifest)
    monkeypatch.setattr(
        stages,
        "open_typed_recovery_gate",
        lambda *args: {
            "gate_sha256": "e" * 64,
            "artifact": {"path": str(exact_receipt)},
        },
    )
    monkeypatch.setattr(stages, "_validate_component_recovery_closure", lambda **kwargs: None)
    view = SimpleNamespace(
        manifest_path=tmp_path / "close.json",
        manifest_sha256="f" * 64,
        pairs_sha256="1" * 64,
        open_vectors=lambda: np.zeros((3, 2), dtype=np.float32),
        open_pairs=lambda: np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )
    monkeypatch.setattr(stages, "validate_theta_close_pair_view", lambda *args, **kwargs: view)

    @contextmanager
    def fake_upstream(path: object):
        del path
        yield {"common_recourse": SimpleNamespace(greedy_counterfactual_summary_from_covering_sets=object())}

    monkeypatch.setattr(stages, "imported_upstream", fake_upstream)
    seen_resume: list[bool] = []

    def fake_summary(*, work_dir: Path, resume: bool, **kwargs: object) -> SimpleNamespace:
        seen_resume.append(resume)
        result_path = Path(work_dir) / "run_manifest.json"
        if not result_path.exists():
            _json(result_path, {"status": "PASS", "run_complete": True})
        return SimpleNamespace(manifest_path=result_path)

    reopens: list[tuple[Path, object, bool]] = []

    def fake_validate(path: Path, *, pair_indices: object, full_replay: bool) -> SimpleNamespace:
        reopens.append((Path(path), pair_indices, full_replay))
        return SimpleNamespace(selected=[], official_result=[])

    monkeypatch.setattr(stages, "summarize_proven_all_core_components_external", fake_summary)
    monkeypatch.setattr(stages, "validate_proven_all_core_component_summary", fake_validate)
    first = stages.run_downstream_stage(
        controller_manifest=manifest_path,
        exact_gate=exact_gate,
        output_dir=output,
        resume=False,
    )
    second = stages.run_downstream_stage(
        controller_manifest=manifest_path,
        exact_gate=exact_gate,
        output_dir=output,
        resume=True,
    )
    assert first == second
    assert seen_resume == [False, True]
    assert reopens == [
        (output / "run_manifest.json", None, True),
        (output / "run_manifest.json", None, True),
    ]


def test_final_stage_reopens_standardized_closure_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "controller/science"
    output.mkdir(parents=True)
    manifest_path = tmp_path / "controller.json"
    bindings = {
        "controller_manifest": manifest_path,
        "adoption_gate": tmp_path / "01.json",
        "subset_gate": tmp_path / "02.json",
        "exact_gate": tmp_path / "03.json",
        "downstream_gate": tmp_path / "04.json",
    }
    terminal = output / "exact_recovery_freeze_receipt.json"
    manifest = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": "a" * 64,
        "resources": {"thread_count": 16},
        "stages": [
            _stage_row(controller.FINAL_STAGE, output, terminal, bindings)
        ],
    }
    monkeypatch.setattr(stages, "load_bound_controller_manifest", lambda path: manifest)
    gate_shas = {
        stage_id: f"{index}" * 64
        for index, stage_id in enumerate(
            (
                controller.ADOPTION_STAGE,
                controller.SUBSET_STAGE,
                controller.EXACT_STAGE,
                controller.DOWNSTREAM_STAGE,
            ),
            start=1,
        )
    }
    monkeypatch.setattr(
        stages,
        "open_typed_recovery_gate",
        lambda value, stage_id: {"gate_sha256": gate_shas[stage_id]},
    )
    calls = 0

    def fake_continuation(inputs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        _json(output / "_RUN_COMPLETE.json", {"status": "PASS", "run_complete": True})
        _json(output / "common_recourse/_RUN_COMPLETE.json", {"run_complete": True})
        _json(output / "standardized/freeze_manifest.json", {"status": "PASS"})
        _file(output / "PASS", b"PASS\n")
        return {"status": "PASS"}

    monkeypatch.setattr(continuation, "run_continuation", fake_continuation)
    common_reopens: list[Path] = []
    monkeypatch.setattr(
        continuation,
        "_validate_common_recourse_completion",
        lambda *, marker, terminal: common_reopens.append(Path(marker)),
    )
    monkeypatch.setattr(stages, "_continuation_inputs", lambda *args: object())
    first = stages.run_final_stage(
        controller_manifest=manifest_path,
        adoption_gate=bindings["adoption_gate"],
        subset_gate=bindings["subset_gate"],
        exact_gate=bindings["exact_gate"],
        downstream_gate=bindings["downstream_gate"],
        output_dir=output,
        resume=True,
    )
    monkeypatch.setattr(stages.os, "getpid", lambda: 5252)
    monkeypatch.setattr(stages.os, "getpgrp", lambda: 5252)
    second = stages.run_final_stage(
        controller_manifest=manifest_path,
        adoption_gate=bindings["adoption_gate"],
        subset_gate=bindings["subset_gate"],
        exact_gate=bindings["exact_gate"],
        downstream_gate=bindings["downstream_gate"],
        output_dir=output,
        resume=True,
    )
    assert first == second
    assert calls == 1
    assert common_reopens == [
        output / "common_recourse/_RUN_COMPLETE.json",
        output / "common_recourse/_RUN_COMPLETE.json",
    ]
