from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from src.chem.hard_deletion import HardDeletionOutcome  # noqa: E402
from src.data.molecular_graph_dataset import (  # noqa: E402
    MolecularGraphData,
    MolecularGraphDataset,
    save_molecular_graph_cache,
)
from src.data.molecular_graph_featurizer import (  # noqa: E402
    default_molecular_feature_schema,
)
from src.eval import tastemolnet_gnn_stages as stages  # noqa: E402
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig  # noqa: E402
from src.oracles.gnn_oracle import classification_metrics  # noqa: E402
from src.utils import tastemolnet_downstream_policy as policy_module  # noqa: E402
from scripts.autodl import tastemolnet_gnn_stage as stage_cli  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_POLICY = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
)
DOWNSTREAM_POLICY = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_output(tmp_path: Path, basename: str) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    return (
        artifact_root,
        artifact_root
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / basename,
    )


def _graph_cache_exclusion(tmp_path: Path) -> Path:
    root = tmp_path / "graph-cache-exclusion"
    root.mkdir(exist_ok=True)
    return root


def _assert_no_usable_stage_pass(output: Path, marker: str) -> None:
    assert output.exists()
    assert not (output / marker).exists()
    with pytest.raises(stages.TasteGNNStageError):
        stages.verify_stage_output(output)
    with pytest.raises(stages.TasteGNNStageError):
        stages.hold_taste_stage_output(output)


def _write_existing_fit(checkpoint: Path, *, temperature: float = 2.0) -> None:
    logits = np.asarray(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [1.5, 2.0, 0.0],
            [0.0, 1.5, 2.0],
            [2.0, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 2, 1, 2, 0], dtype=np.int64)
    raw = stages._softmax(logits, temperature=1.0)
    stored_raw = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=1
    ).numpy()
    calibrated = stages._softmax(logits, temperature=temperature)
    before = classification_metrics(labels, raw, num_classes=3)
    after = classification_metrics(labels, calibrated, num_classes=3)
    payload = {
        "schema_version": "temperature_scaling_v1",
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "temperature": temperature,
        "num_examples": len(labels),
        "num_classes": 3,
        "nll_before": stages._nll(raw, labels),
        "nll_after": stages._nll(calibrated, labels),
        "ece_before": before["ece"],
        "ece_after": after["ece"],
        "brier_before": before["brier_score"],
        "brier_after": after["brier_score"],
        "argmax_invariant": True,
    }
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with (checkpoint / "validation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "molecule_id",
                "smiles",
                "split",
                "label",
                "predicted_label",
                "logits",
                "probabilities",
                "source_graph_hash",
            ],
        )
        writer.writeheader()
        for index, (label, row_logits, row_probs) in enumerate(
            zip(labels, logits, stored_raw, strict=True)
        ):
            writer.writerow(
                {
                    "molecule_id": f"validation-{index}",
                    "smiles": "CC",
                    "split": "val",
                    "label": int(label),
                    "predicted_label": int(np.argmax(row_probs)),
                    "logits": json.dumps(row_logits.tolist()),
                    "probabilities": json.dumps(row_probs.tolist()),
                    "source_graph_hash": f"{index:064x}",
                }
            )
    (checkpoint / "split_manifest.json").write_text(
        json.dumps(
            {
                "roles": {
                    "validation": "checkpoint_selection_and_temperature_calibration"
                },
                "validation_manifest": {"num_records": len(labels)},
                "calibration_loaded_for_training": False,
                "test_loaded_for_training": False,
                "test_evaluated_during_training": False,
                "test_used_for_checkpoint_selection": False,
            }
        ),
        encoding="utf-8",
    )


def _model_card(model_sha: str) -> dict[str, Any]:
    return {
        "dataset": "tastemolnet",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "profile": "full",
        "selection_split": "validation",
        "temperature_calibration_split": "validation",
        "temperature_calibration_fit_on_validation": True,
        "calibration_used_for_model_fit_or_selection": False,
        "test_used_for_model_fit_or_selection": False,
        "test_loaded_during_training": False,
        "test_evaluated_during_training": False,
        "graph_cache_used": True,
        "num_classes": 3,
        "source_label": 1,
        "health_gate": {"status": "PASS"},
        "checkpoint_id": model_sha,
        "graph_cache_manifest_sha256": "b" * 64,
        "data_use_policy_file_sha256": "c" * 64,
        "data_use_policy_canonical_sha256": "d" * 64,
        "data_use_policy_receipt_sha256": "e" * 64,
    }


def _minimal_t2_bundle(root: Path) -> dict[str, Any]:
    root.mkdir()
    (root / "model.pt").write_bytes(b"selected-best-model")
    (root / "last.pt").write_bytes(b"terminal-latest-model")
    for name, value in (
        ("config.yaml", "{}\n"),
        (
            "feature_schema.json",
            json.dumps(default_molecular_feature_schema().to_dict()),
        ),
    ):
        (root / name).write_text(value, encoding="utf-8")
    (root / "label_map.json").write_text(json.dumps(stages.LABEL_MAP), encoding="utf-8")
    (root / "test_evaluation_status.json").write_text(
        json.dumps(
            {
                "status": "NOT_EVALUATED",
                "test_loaded": False,
                "reason": "held_out_until_frozen_final_evaluation",
                "path": "/private/tastemolnet/test.csv",
                "sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    _write_existing_fit(root)
    card = _model_card(_sha(root / "model.pt"))
    (root / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    split_manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    split_manifest["files"] = {
        "test": {"path": "/private/tastemolnet/test.csv", "sha256": "f" * 64}
    }
    (root / "split_manifest.json").write_text(
        json.dumps(split_manifest), encoding="utf-8"
    )
    last_checkpoint = {
        "schema_version": "tastemolnet_last_training_checkpoint_v1",
        "checkpoint_file": "last.pt",
        "same_bytes_as_latest_epoch_checkpoint": True,
        "completed_epoch": 42,
        "checkpoint_sha256": _sha(root / "last.pt"),
        "source_checkpoint_sha256": _sha(root / "last.pt"),
    }
    payloads = {
        "data_use_policy_binding.json": {
            "schema_version": "tastemolnet_training_policy_binding_v1",
            "dataset": "tastemolnet",
            "status": "NOT_EXPLICITLY_STATED",
            "authorization_status": "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION",
            "paper_result_reporting_allowed": True,
            "dataset_redistributed": False,
            "data_redistribution_allowed": False,
            "upstream_license_not_explicit": True,
            "upstream_license_status": "NOT_EXPLICITLY_STATED",
            "upstream_license_claimed_resolved": False,
            "license_pass_claimed": False,
            "hpc_execution_authorized": False,
            "policy": {
                "policy_file_sha256": "c" * 64,
                "policy_canonical_sha256": "d" * 64,
            },
            "policy_receipt": {"sha256": "e" * 64},
        },
        "graph_cache_usage.json": {
            "schema_version": "tastemolnet_graph_cache_usage_v1",
            "dataset": "tastemolnet",
            "mode": "read_only_existing_cache",
            "graph_cache_used": True,
            "loaded_splits": ["train", "validation"],
            "calibration_loaded": False,
            "test_loaded": False,
            "graph_cache_rebuilt": False,
            "data_reprepared": False,
            "graph_cache_manifest_sha256": "b" * 64,
        },
        "oracle_manifest.json": {
            "schema_version": "tastemolnet_three_class_gine_oracle_manifest_v1",
            "dataset": "tastemolnet",
            "status": "PASS",
            "checkpoint_id": card["checkpoint_id"],
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "num_classes": 3,
            "source_label": 1,
            "test_loaded": False,
            "test_evaluated": False,
            "paper_result_reporting_allowed": True,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "health_gate": {"status": "PASS"},
        },
        "last_checkpoint.json": last_checkpoint,
        "checkpoint_reload.json": {
            "schema_version": "tastemolnet_gine_checkpoint_reload_v1",
            "status": "PASS",
            "checkpoint_reload_pass": True,
            "batch_single_probability_equivalence": True,
            "all_probabilities_finite": True,
            "num_classes": 3,
            "source_label": 1,
            "checkpoint_id": card["checkpoint_id"],
            "last_checkpoint": last_checkpoint,
        },
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    for name in ("training_metrics.json", "environment.json", "git_state.json"):
        (root / name).write_text("{}\n", encoding="utf-8")
    lines = [
        f"{_sha(path)}  {path.name}"
        for path in sorted(root.iterdir())
        if path.name != "sha256sums.txt"
    ]
    (root / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card


def test_t3_adopts_existing_fit_without_refit_copy_or_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)

    import src.oracles.gnn_oracle as oracle_module

    monkeypatch.setattr(
        oracle_module,
        "fit_temperature_scaling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("refit called")),
    )
    before = {path.name: _sha(path) for path in checkpoint.iterdir()}
    artifact_root, output = _stage_output(tmp_path, "calibrated-fixture")
    result = stages.run_t3_existing_fit_adoption(
        checkpoint_dir=checkpoint,
        graph_cache_root=_graph_cache_exclusion(tmp_path),
        artifact_root=artifact_root,
        output_dir=output,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
    )
    after = {path.name: _sha(path) for path in checkpoint.iterdir()}

    assert result["status"] == "PASS"
    assert result["calibration"]["existing_fit_adopted"] is True
    assert result["calibration"]["temperature_refit_performed"] is False
    assert result["checkpoint_copied"] is False
    assert result["bundle_evidence_files_opened"] == ["validation_predictions.csv"]
    assert result["external_split_payload_files_opened"] == []
    assert before == after
    assert not (output / "model.pt").exists()
    assert not list(output.glob("*.csv"))
    assert not (output / "predictions.csv").exists()
    assert stages.verify_stage_output(output)["gate"]["status"] == "PASS"
    with stages.hold_taste_stage_output(output) as held:
        assert set(held.evidence) == {
            "stage",
            "gate_sha256",
            "root_inventory_sha256",
            "checkpoint_dir",
            "checkpoint_id",
            "checkpoint_inventory_sha256",
            "checkpoint_stat_inventory_sha256",
            "checkpoint_sha256s_sha256",
        }
        assert held.evidence["stage"] == stages.T3_STAGE
        assert held.evidence["checkpoint_id"] == result["checkpoint_id"]
        assert held.revalidate() == dict(held.evidence)
        with stages.hold_taste_checkpoint_bundle(
            checkpoint,
            expected_stage_evidence=held.evidence,
        ) as held_checkpoint:
            assert held_checkpoint.checkpoint_dir == checkpoint
            assert held_checkpoint.revalidate() == dict(held.evidence)
            assert held_checkpoint.read_frozen_gine_payload("model.pt") == (
                checkpoint / "model.pt"
            ).read_bytes()
            assert held_checkpoint.read_frozen_gine_payload(
                "split_manifest.json"
            ) == (checkpoint / "split_manifest.json").read_bytes()
            test_status = json.loads(
                held_checkpoint.read_frozen_gine_payload(
                    "test_evaluation_status.json"
                ).decode("utf-8")
            )
            assert test_status["status"] == "NOT_EVALUATED"
            assert test_status["test_loaded"] is False
            with pytest.raises(stages.TasteGNNStageError, match="may not open"):
                held_checkpoint.read_frozen_gine_payload(
                    "validation_predictions.csv"
                )
            for rejected_name in (
                "training_metrics.json",
                "last.pt",
                "train.csv",
                "validation.csv",
                "calibration.csv",
                "test.csv",
            ):
                with pytest.raises(stages.TasteGNNStageError, match="may not open"):
                    held_checkpoint.read_frozen_gine_payload(rejected_name)


def test_t3_rejects_old_checkpoint_descendant_output_exploit_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    output = (
        checkpoint
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / "calibrated-exploit"
    )
    before = {path.name: _sha(path) for path in checkpoint.iterdir()}
    with pytest.raises(stages.TasteGNNStageError, match="overlaps protected"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=checkpoint,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    assert not (checkpoint / "gnn_oracles").exists()
    assert before == {path.name: _sha(path) for path in checkpoint.iterdir()}


def test_t3_rejects_graph_cache_descendant_output_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    cache_root = tmp_path / "graph-cache"
    cache_root.mkdir()
    output = (
        cache_root
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / "calibrated-cache-exploit"
    )
    with pytest.raises(stages.TasteGNNStageError, match="overlaps protected"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=cache_root,
            artifact_root=cache_root,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    assert not (cache_root / "gnn_oracles").exists()


def test_held_checkpoint_rejects_copy_symlink_alias_and_late_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-checkpoint-hold")
    stages.run_t3_existing_fit_adoption(
        checkpoint_dir=checkpoint,
        graph_cache_root=_graph_cache_exclusion(tmp_path),
        artifact_root=artifact_root,
        output_dir=output,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
    )
    copied = tmp_path / "copied-t2"
    shutil.copytree(checkpoint, copied)
    alias = tmp_path / "aliased-t2"
    alias.symlink_to(checkpoint, target_is_directory=True)
    with stages.hold_taste_stage_output(output) as held_stage:
        for rejected in (copied, alias):
            with pytest.raises(stages.TasteGNNStageError, match="exact stage authority"):
                stages.hold_taste_checkpoint_bundle(
                    rejected,
                    expected_stage_evidence=held_stage.evidence,
                )
        held_checkpoint = stages.hold_taste_checkpoint_bundle(
            checkpoint,
            expected_stage_evidence=held_stage.evidence,
        )
        (checkpoint / "model.pt").write_bytes(b"late-model-mutation")
        try:
            with pytest.raises(stages.TasteGNNStageError, match="authority changed"):
                held_checkpoint.revalidate()
        finally:
            held_checkpoint.close()


def test_t3_rejects_output_outside_exact_artifact_formula_before_creation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    invalid = tmp_path / "calibrated-outside"
    with pytest.raises(stages.TasteGNNStageError, match="direct child"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=artifact_root,
            output_dir=invalid,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    assert not invalid.exists()
    assert not (artifact_root / "gnn_oracles").exists()


def test_output_creation_rejects_artifact_root_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-root-swap")
    alternate = tmp_path / "alternate-artifacts"
    alternate.mkdir()
    parked = tmp_path / "parked-artifacts"
    real_mkdir = os.mkdir
    triggered = False

    def swapping_mkdir(
        path: Any,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal triggered
        if path == "gnn_oracles" and dir_fd is not None and not triggered:
            triggered = True
            artifact_root.rename(parked)
            alternate.rename(artifact_root)
            try:
                real_mkdir(path, mode, dir_fd=dir_fd)
            finally:
                artifact_root.rename(alternate)
                parked.rename(artifact_root)
            return
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stages.os, "mkdir", swapping_mkdir)
    with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=artifact_root,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    assert triggered is True
    assert not (output / stages.T3_MARKER).exists()


def test_output_creation_rejects_output_parent_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-parent-swap")
    parent = output.parent
    parent.mkdir(parents=True)
    alternate = parent.parent / "alternate-seed7"
    alternate.mkdir()
    parked = parent.parent / "parked-seed7"
    real_mkdir = os.mkdir
    triggered = False

    def swapping_mkdir(
        path: Any,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal triggered
        if path == output.name and dir_fd is not None and not triggered:
            triggered = True
            parent.rename(parked)
            alternate.rename(parent)
            try:
                real_mkdir(path, mode, dir_fd=dir_fd)
            finally:
                parent.rename(alternate)
                parked.rename(parent)
            return
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stages.os, "mkdir", swapping_mkdir)
    with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=artifact_root,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    assert triggered is True
    assert not (output / stages.T3_MARKER).exists()


def test_t3_publication_time_model_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-model-mutation")
    real_prepare = stages._prepare_stage_output

    def injecting_prepare(*args: Any, **kwargs: Any) -> dict[str, dict[str, int]]:
        publication = real_prepare(*args, **kwargs)
        (checkpoint / "model.pt").write_bytes(b"publication-time-mutation")
        return publication

    monkeypatch.setattr(stages, "_prepare_stage_output", injecting_prepare)
    with pytest.raises(stages.TasteGNNStageError, match="checkpoint bytes drifted"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=artifact_root,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    _assert_no_usable_stage_pass(output, stages.T3_MARKER)


def test_t3_marker_publish_window_model_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-marker-window")
    real_publish = stages._publish_prepared_stage_marker

    def injecting_publish(
        prepared: stages.PreparedStageOutput,
        *,
        retained_input_closure: Any,
    ) -> dict[str, int]:
        (checkpoint / "model.pt").write_bytes(b"marker-window-mutation")
        return real_publish(
            prepared,
            retained_input_closure=retained_input_closure,
        )

    monkeypatch.setattr(stages, "_publish_prepared_stage_marker", injecting_publish)
    with pytest.raises(stages.TasteGNNStageError, match="checkpoint bytes drifted"):
        stages.run_t3_existing_fit_adoption(
            checkpoint_dir=checkpoint,
            graph_cache_root=_graph_cache_exclusion(tmp_path),
            artifact_root=artifact_root,
            output_dir=output,
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
        )
    _assert_no_usable_stage_pass(output, stages.T3_MARKER)


@pytest.mark.parametrize("policy_path", [DOWNSTREAM_POLICY, BASE_POLICY])
def test_t3_publication_time_policy_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_path: Path
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-policy-mutation")
    original = policy_path.read_bytes()
    real_prepare = stages._prepare_stage_output

    def injecting_prepare(*args: Any, **kwargs: Any) -> dict[str, dict[str, int]]:
        publication = real_prepare(*args, **kwargs)
        policy_path.write_bytes(original + b"\n")
        return publication

    monkeypatch.setattr(stages, "_prepare_stage_output", injecting_prepare)
    try:
        with pytest.raises(
            policy_module.TasteDownstreamPolicyError,
            match="file authority changed|bytes changed",
        ):
            stages.run_t3_existing_fit_adoption(
                checkpoint_dir=checkpoint,
                graph_cache_root=_graph_cache_exclusion(tmp_path),
                artifact_root=artifact_root,
                output_dir=output,
                downstream_policy_path=DOWNSTREAM_POLICY,
                base_policy_path=BASE_POLICY,
            )
    finally:
        policy_path.write_bytes(original)
    _assert_no_usable_stage_pass(output, stages.T3_MARKER)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "not_fit", "unsafe"),
        ("test_used_for_fit", True, "unsafe"),
        ("argmax_invariant", False, "unsafe"),
        ("temperature", 0.0, "outside"),
        ("num_classes", True, "native JSON integer"),
        ("nll_after", 999.0, "differs"),
    ],
)
def test_t3_existing_fit_rejects_unsafe_or_tampered_evidence(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    path = checkpoint / "temperature_scaling.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(stages.TasteGNNStageError, match=message):
        stages.validate_existing_temperature_fit(checkpoint)


def test_taste_model_card_rejects_equal_bool_and_float_class_values() -> None:
    card = _model_card("0" * 64)
    stages.validate_taste_model_card(card)
    for field, value in (("num_classes", 3.0), ("source_label", True)):
        mutated = {**card, field: value}
        with pytest.raises(stages.TasteGNNStageError, match="model-card contract"):
            stages.validate_taste_model_card(mutated)


def test_cli_requires_the_exact_tracked_hpc_config(tmp_path: Path) -> None:
    hpc = PROJECT_ROOT / "configs/hpc.yaml"
    stage_cli._validate_configs([str(hpc)])
    copied = tmp_path / "hpc.yaml"
    copied.write_bytes(hpc.read_bytes())
    for values in ([], [str(copied)], [str(hpc), str(hpc)]):
        with pytest.raises(ValueError, match="config"):
            stage_cli._validate_configs(values)


def test_physical_directory_detects_named_root_replacement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    held = stages._physical_directory(root, field="fixture root")
    moved = tmp_path / "moved"
    root.rename(moved)
    root.mkdir()
    try:
        with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
            held.verify(label="fixture root")
    finally:
        held.close()


def test_temperature_openat_rejects_temporary_root_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _minimal_t2_bundle(checkpoint)
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "temperature_scaling.json").write_text(
        json.dumps({"temperature": 999.0}), encoding="utf-8"
    )
    parked = tmp_path / "parked"
    real_open = os.open
    triggered = False

    def swapping_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal triggered
        if path == "temperature_scaling.json" and dir_fd is not None and not triggered:
            triggered = True
            checkpoint.rename(parked)
            alternate.rename(checkpoint)
            try:
                return real_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                checkpoint.rename(alternate)
                parked.rename(checkpoint)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stages.os, "open", swapping_open)
    with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
        stages.validate_existing_temperature_fit(checkpoint)
    assert triggered is True
    assert json.loads(
        (checkpoint / "temperature_scaling.json").read_text(encoding="utf-8")
    )["temperature"] == 2.0


def test_oracle_model_openat_rejects_temporary_root_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = default_molecular_feature_schema()
    config = MolecularGNNConfig(
        backbone="gine",
        num_classes=3,
        num_layers=1,
        hidden_dim=8,
        dropout=0.0,
        pooling="mean",
        readout_layers=1,
        normalization="layer_norm",
        residual=True,
    )
    model = MolecularGNN(
        config,
        node_cardinalities=schema.node_cardinalities,
        edge_cardinalities=schema.edge_cardinalities,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {
            "bundle_version": stages.CHECKPOINT_BUNDLE_VERSION,
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "feature_schema_sha256": schema.to_dict()["schema_sha256"],
        },
        checkpoint / "model.pt",
    )
    (checkpoint / "model_card.json").write_text(
        json.dumps({"backbone": "gine", "num_classes": 3, "source_label": 1}),
        encoding="utf-8",
    )
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps({"temperature": 2.0}), encoding="utf-8"
    )
    expected_checkpoint_id = _sha(checkpoint / "model.pt")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "model.pt").write_bytes(b"malicious-alternate-model")
    parked = tmp_path / "parked"
    held = stages._physical_directory(checkpoint, field="checkpoint")
    real_open = os.open
    triggered = False

    def swapping_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal triggered
        if path == "model.pt" and dir_fd is not None and not triggered:
            triggered = True
            checkpoint.rename(parked)
            alternate.rename(checkpoint)
            try:
                return real_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                checkpoint.rename(alternate)
                parked.rename(checkpoint)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stages.os, "open", swapping_open)
    try:
        with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
            stages._load_gnn_oracle_anchored(
                held,
                feature_schema=schema,
                device="cpu",
                batch_size=4,
            )
    finally:
        held.close()
    assert triggered is True
    assert _sha(checkpoint / "model.pt") == expected_checkpoint_id


def test_stage_output_verifier_rejects_temporary_root_swap_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    held_output = stages._physical_directory(output, field="stage output")
    stages._write_stage_output(
        held_output,
        documents={
            "gate.json": {
                "status": "PASS",
                "marker": "EXPECTED_PASS",
            }
        },
        marker="EXPECTED_PASS",
    )
    held_output.close()
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "gate.json").write_text(
        json.dumps({"status": "PASS", "marker": "FORGED_PASS"}),
        encoding="utf-8",
    )
    parked = tmp_path / "parked"
    real_open = os.open
    triggered = False

    def swapping_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal triggered
        if path == "gate.json" and dir_fd is not None and not triggered:
            triggered = True
            output.rename(parked)
            alternate.rename(output)
            try:
                return real_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                output.rename(alternate)
                parked.rename(output)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stages.os, "open", swapping_open)
    with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
        stages.verify_stage_output(output)
    assert triggered is True
    assert json.loads((output / "gate.json").read_text(encoding="utf-8"))["marker"] == (
        "EXPECTED_PASS"
    )


def test_stage_publication_rejects_injected_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    held = stages._physical_directory(output, field="stage output")
    real_write = stages._write_bytes_at
    triggered = False

    def injecting_write(*args: Any, **kwargs: Any) -> None:
        nonlocal triggered
        real_write(*args, **kwargs)
        if not triggered:
            triggered = True
            (output / "injected.txt").write_text("forbidden\n", encoding="utf-8")

    monkeypatch.setattr(stages, "_write_bytes_at", injecting_write)
    try:
        with pytest.raises(stages.TasteGNNStageError, match="unexpected file"):
            stages._write_stage_output(
                held,
                documents={
                    "gate.json": {"status": "PASS", "marker": "EXPECTED_PASS"}
                },
                marker="EXPECTED_PASS",
            )
    finally:
        held.close()
    assert triggered is True
    assert not (output / "sha256sums.txt").exists()


def test_prepared_stage_is_not_pass_until_marker_is_committed_last(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    held = stages._physical_directory(output, field="stage output")
    try:
        publication = stages._prepare_stage_output(
            held,
            documents={
                "gate.json": {"status": "PASS", "marker": "EXPECTED_PASS"}
            },
            marker="EXPECTED_PASS",
        )
        assert publication.sha_identity
        assert not (output / "EXPECTED_PASS").exists()
        with pytest.raises(stages.TasteGNNStageError):
            stages.verify_stage_output(held)
        publication.revalidate()
        stages._publish_prepared_stage_marker(
            publication,
            retained_input_closure=lambda: None,
        )
        assert stages.verify_stage_output(held)["gate"]["status"] == "PASS"
        publication.close()
    finally:
        held.close()


@pytest.mark.parametrize("name", ("gate.json", "sha256sums.txt"))
def test_prepared_stage_rejects_equal_byte_inode_replacement_before_commit(
    tmp_path: Path,
    name: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    held = stages._physical_directory(output, field="stage output")
    prepared = stages._prepare_stage_output(
        held,
        documents={
            "gate.json": {"status": "PASS", "marker": "EXPECTED_PASS"}
        },
        marker="EXPECTED_PASS",
    )
    try:
        target = output / name
        original = tmp_path / f"original-{name}"
        data = target.read_bytes()
        target.rename(original)
        target.write_bytes(data)
        with pytest.raises(
            stages.TasteGNNStageError,
            match="identity drifted|retained prepared stage file changed",
        ):
            prepared.revalidate()
        assert not (output / "EXPECTED_PASS").exists()
        with pytest.raises(stages.TasteGNNStageError):
            stages.verify_stage_output(held)
    finally:
        prepared.close()
        held.close()


@pytest.mark.parametrize("name", ("gate.json", "sha256sums.txt"))
def test_terminal_marker_physically_binds_prepared_file_inodes(
    tmp_path: Path,
    name: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    held = stages._physical_directory(output, field="stage output")
    try:
        stages._write_stage_output(
            held,
            documents={
                "gate.json": {"status": "PASS", "marker": "EXPECTED_PASS"}
            },
            marker="EXPECTED_PASS",
        )
        assert stages.verify_stage_output(held)["gate"]["status"] == "PASS"
        held.close()
        target = output / name
        original = tmp_path / f"original-{name}"
        data = target.read_bytes()
        target.rename(original)
        target.write_bytes(data)
        with pytest.raises(
            stages.TasteGNNStageError,
            match="physically unbound",
        ):
            stages.verify_stage_output(output)
    finally:
        held.close()


def test_held_stage_output_rejects_swap_restore_across_consumer_window(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "t2"
    _minimal_t2_bundle(checkpoint)
    artifact_root, output = _stage_output(tmp_path, "calibrated-held")
    stages.run_t3_existing_fit_adoption(
        checkpoint_dir=checkpoint,
        graph_cache_root=_graph_cache_exclusion(tmp_path),
        artifact_root=artifact_root,
        output_dir=output,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
    )
    held = stages.hold_taste_stage_output(output)
    parked = output.parent / "parked-held"
    alternate = output.parent / "alternate-held"
    alternate.mkdir()
    output.rename(parked)
    alternate.rename(output)
    output.rename(alternate)
    parked.rename(output)
    try:
        with pytest.raises(stages.TasteGNNStageError, match="identity drifted"):
            held.revalidate()
    finally:
        held.close()


def test_t4_cache_loader_opens_only_manifest_and_calibration_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "calibration.csv"
    source.write_text(
        "molecule_id,smiles,label,split\n"
        "c0,CC,0,calibration\n"
        "c1,CN,1,calibration\n"
        "c2,CO,2,calibration\n",
        encoding="utf-8",
    )
    dataset = MolecularGraphDataset.from_csv(
        source, num_classes=3, expected_split="calibration"
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    entry = save_molecular_graph_cache(
        dataset, cache_root / "calibration.pt", split_name="calibration"
    )
    manifest = {
        "schema_version": "molecular_graph_cache_manifest_v1",
        "dataset": "tastemolnet",
        "num_classes": 3,
        "split_order": ["train", "validation", "calibration", "test"],
        "splits": {
            "train": {},
            "validation": {},
            "calibration": {
                "cache_file": "calibration.pt",
                "cache_sha256": entry["sha256"],
                "source_csv_sha256": entry["source_csv_sha256"],
                "graph_count": entry["graph_count"],
                "num_classes": 3,
                "safe_load_verified": True,
            },
            "test": {"cache_file": "test.pt", "cache_sha256": "f" * 64},
        },
    }
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (cache_root / "test.pt").write_bytes(b"must-never-be-opened")
    (cache_root / "train.pt").write_bytes(b"must-never-be-opened")
    (cache_root / "validation.pt").write_bytes(b"must-never-be-opened")
    real_open = os.open
    opened: list[str] = []

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        value = Path(path)
        if value.name in {
            "manifest.json",
            "calibration.pt",
            "train.pt",
            "validation.pt",
            "test.pt",
        }:
            opened.append(value.name)
            if value.name in {"train.pt", "validation.pt", "test.pt"}:
                raise AssertionError(f"forbidden split opened: {value.name}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(stages.os, "open", guarded_open)
    loaded, evidence = stages._load_calibration_cache(
        graph_cache_root=cache_root,
        expected_manifest_sha256=_sha(manifest_path),
        feature_schema=dataset.feature_schema,
    )
    assert len(loaded) == 3
    assert opened == ["manifest.json", "calibration.pt"]
    assert evidence["opened_payload_splits"] == ["calibration"]
    assert evidence["test_payload_opened"] is False
    assert evidence["csv_payload_opened"] is False


def _graph(index: int, *, residual: bool = False) -> MolecularGraphData:
    return MolecularGraphData(
        x=((0,),),
        edge_index=((), ()),
        edge_attr=(),
        y=-1 if residual else 1,
        molecule_id=(f"residual-{index}" if residual else f"parent-{index}"),
        smiles="C" if residual else "CC",
        split="oracle_smoke" if residual else "calibration",
        graph_sha256=f"{index + 1:064x}",
    )


def _outcome(index: int) -> HardDeletionOutcome:
    return HardDeletionOutcome(
        parent_id=f"parent-{index}",
        candidate_id=f"candidate-{index}",
        match_id=0,
        match_atom_indices=(0,),
        removed_atom_symbols=("C",),
        removed_atom_count=1,
        removed_bond_count=1,
        boundary_bond_count=1,
        residual_smiles="C",
        residual_heavy_atom_count=1,
        residual_num_components=1,
        residual_connected=True,
        sanitize_ok=True,
        contains_dot=False,
        valid=True,
        invalid_reason=None,
        atom_delete_ratio=0.5,
        bond_delete_ratio=1.0,
        residual_atom_count=1,
        residual_bond_count=0,
    )


class _FakeDataset:
    def __init__(self, count: int = 18) -> None:
        self.values = [_graph(index) for index in range(count)]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> MolecularGraphData:
        return self.values[index]


class _FakeOracle:
    checkpoint_id = "a" * 64
    backbone = "gine"
    num_classes = 3
    source_label = 1
    temperature = 2.0

    @staticmethod
    def _record(graph: MolecularGraphData) -> dict[str, Any]:
        if graph.molecule_id.startswith("t4-residual-"):
            position = int(graph.molecule_id.split("-")[2])
            after = (0, 2, 1)[position % 3]
            probabilities = {
                0: [0.80, 0.10, 0.10],
                1: [0.10, 0.80, 0.10],
                2: [0.10, 0.10, 0.80],
            }[after]
        else:
            after = 1
            probabilities = [0.10, 0.80, 0.10]
        return {
            "predicted_label": after,
            "probabilities": probabilities,
            "logits": [0.0, 1.0, 0.0],
            "source_probability": probabilities[1],
            "confidence": max(probabilities),
            "checkpoint_id": _FakeOracle.checkpoint_id,
            "backbone": "gine",
            "num_classes": 3,
            "temperature": _FakeOracle.temperature,
            "source_label": 1,
        }

    def predict_records(self, graphs: Any, *, batch_size: int) -> list[dict[str, Any]]:
        return [self._record(graph) for graph in graphs]

    def predict_proba(self, graphs: Any, *, batch_size: int) -> np.ndarray:
        return np.asarray([self._record(graph)["probabilities"] for graph in graphs])


def _fake_t4_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    checkpoint = tmp_path / "t2"
    card = _minimal_t2_bundle(checkpoint)
    artifact_root, t3 = _stage_output(tmp_path, "calibrated-for-t4")
    stages.run_t3_existing_fit_adoption(
        checkpoint_dir=checkpoint,
        graph_cache_root=_graph_cache_exclusion(tmp_path),
        artifact_root=artifact_root,
        output_dir=t3,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (cache_root / "calibration.pt").write_bytes(b"fixture-calibration-cache")
    monkeypatch.setattr(
        stages,
        "_load_calibration_cache",
        lambda **_kwargs: (
            _FakeDataset(),
            {
                "opened_payload_splits": ["calibration"],
                "csv_payload_opened": False,
                "test_payload_opened": False,
            },
        ),
    )
    monkeypatch.setattr(
        stages,
        "run_bounded_oracle_smoke",
        lambda **_kwargs: {"status": "PASS", "selected_count": 16},
    )

    def fake_factory(_checkpoint: stages.PhysicalDirectory, **_kwargs: Any) -> _FakeOracle:
        oracle = _FakeOracle()
        oracle.checkpoint_id = card["checkpoint_id"]
        return oracle

    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    monkeypatch.setenv("AUTODL_PHYSICAL_GPU_INDEX", "1")
    monkeypatch.setenv("AUTODL_PHYSICAL_GPU_UUID", gpu_uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    output = (
        artifact_root
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / "t4-oracle-smoke-hostile"
    )
    kwargs = {
        "checkpoint_dir": checkpoint,
        "t3_gate_path": t3 / "gate.json",
        "graph_cache_root": cache_root,
        "artifact_root": artifact_root,
        "output_dir": output,
        "downstream_policy_path": DOWNSTREAM_POLICY,
        "base_policy_path": BASE_POLICY,
        "gpu_uuid": gpu_uuid,
        "oracle_factory": fake_factory,
    }
    return kwargs, checkpoint, cache_root, t3, output


def test_t4_does_not_open_checkpoint_validation_csv_and_binds_exp_run_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "t2"
    card = _minimal_t2_bundle(checkpoint)
    artifact_root, t3 = _stage_output(tmp_path, "calibrated-fixture")
    stages.run_t3_existing_fit_adoption(
        checkpoint_dir=checkpoint,
        graph_cache_root=_graph_cache_exclusion(tmp_path),
        artifact_root=artifact_root,
        output_dir=t3,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
    )
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (cache_root / "calibration.pt").write_bytes(b"fixture-calibration-cache")
    monkeypatch.setattr(
        stages,
        "_load_calibration_cache",
        lambda **_kwargs: (
            _FakeDataset(),
            {
                "opened_payload_splits": ["calibration"],
                "csv_payload_opened": False,
                "test_payload_opened": False,
            },
        ),
    )
    monkeypatch.setattr(
        stages,
        "run_bounded_oracle_smoke",
        lambda **_kwargs: {"status": "PASS", "selected_count": 16},
    )

    def fake_factory(_checkpoint: Path, **kwargs: Any) -> _FakeOracle:
        assert kwargs["verify_hashes"] is False
        assert kwargs["require_taste_closure"] is True
        assert kwargs["device"] == "cuda:0"
        oracle = _FakeOracle()
        oracle.checkpoint_id = card["checkpoint_id"]
        return oracle

    real_read_bytes_at = stages._read_bytes_at
    opened: list[str] = []

    def guarded_read_bytes_at(
        directory: stages.PhysicalDirectory,
        name: str,
        **kwargs: Any,
    ) -> tuple[bytes, dict[str, Any]]:
        if name == "validation_predictions.csv":
            raise AssertionError("T4 reopened validation_predictions.csv")
        opened.append(name)
        return real_read_bytes_at(directory, name, **kwargs)

    monkeypatch.setattr(stages, "_read_bytes_at", guarded_read_bytes_at)
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    monkeypatch.setenv("AUTODL_PHYSICAL_GPU_INDEX", "1")
    monkeypatch.setenv("AUTODL_PHYSICAL_GPU_UUID", gpu_uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    output = (
        artifact_root
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / "t4-oracle-smoke-fixture"
    )
    result = stages.run_t4_calibration_cache_smoke(
        checkpoint_dir=checkpoint,
        t3_gate_path=t3 / "gate.json",
        graph_cache_root=cache_root,
        artifact_root=artifact_root,
        output_dir=output,
        downstream_policy_path=DOWNSTREAM_POLICY,
        base_policy_path=BASE_POLICY,
        gpu_uuid=gpu_uuid,
        oracle_factory=fake_factory,
    )
    assert result["status"] == "PASS"
    assert "validation_predictions.csv" not in opened
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["physical_gpu_index"] == 1
    assert gate["gpu_uuid"] == gpu_uuid
    assert gate["cuda_visible_devices"] == "1"


def test_t4_rejects_old_graph_cache_descendant_output_exploit_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _checkpoint, cache_root, _t3, _output = _fake_t4_route(
        tmp_path, monkeypatch
    )
    exploit = (
        cache_root
        / "gnn_oracles"
        / "tastemolnet"
        / "gine"
        / "seed7"
        / "t4-oracle-smoke-exploit"
    )
    before = {path.name: _sha(path) for path in cache_root.iterdir()}
    kwargs["artifact_root"] = cache_root
    kwargs["output_dir"] = exploit
    with pytest.raises(stages.TasteGNNStageError, match="overlaps protected"):
        stages.run_t4_calibration_cache_smoke(**kwargs)
    assert not (cache_root / "gnn_oracles").exists()
    assert before == {path.name: _sha(path) for path in cache_root.iterdir()}


def test_t4_publication_time_calibration_cache_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _checkpoint, cache_root, _t3, output = _fake_t4_route(
        tmp_path, monkeypatch
    )
    real_prepare = stages._prepare_stage_output

    def injecting_prepare(
        *args: Any, **publish_kwargs: Any
    ) -> dict[str, dict[str, int]]:
        publication = real_prepare(*args, **publish_kwargs)
        (cache_root / "calibration.pt").write_bytes(b"publication-time-mutation")
        return publication

    monkeypatch.setattr(stages, "_prepare_stage_output", injecting_prepare)
    with pytest.raises(stages.TasteGNNStageError, match="calibration-cache authority drifted"):
        stages.run_t4_calibration_cache_smoke(**kwargs)
    _assert_no_usable_stage_pass(output, stages.T4_MARKER)


def test_t4_marker_publish_window_cache_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _checkpoint, cache_root, _t3, output = _fake_t4_route(
        tmp_path, monkeypatch
    )
    real_publish = stages._publish_prepared_stage_marker

    def injecting_publish(
        prepared: stages.PreparedStageOutput,
        *,
        retained_input_closure: Any,
    ) -> dict[str, int]:
        (cache_root / "calibration.pt").write_bytes(b"marker-window-mutation")
        return real_publish(
            prepared,
            retained_input_closure=retained_input_closure,
        )

    monkeypatch.setattr(stages, "_publish_prepared_stage_marker", injecting_publish)
    with pytest.raises(
        stages.TasteGNNStageError,
        match="calibration-cache authority drifted",
    ):
        stages.run_t4_calibration_cache_smoke(**kwargs)
    _assert_no_usable_stage_pass(output, stages.T4_MARKER)


def test_t4_publication_time_t3_gate_mutation_never_returns_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _checkpoint, _cache_root, t3, output = _fake_t4_route(
        tmp_path, monkeypatch
    )
    real_prepare = stages._prepare_stage_output

    def injecting_prepare(
        *args: Any, **publish_kwargs: Any
    ) -> dict[str, dict[str, int]]:
        publication = real_prepare(*args, **publish_kwargs)
        (t3 / "gate.json").write_text("{}\n", encoding="utf-8")
        return publication

    monkeypatch.setattr(stages, "_prepare_stage_output", injecting_prepare)
    with pytest.raises(
        stages.TasteGNNStageError,
        match="stage output hash mismatch|complete T3 output drifted",
    ):
        stages.run_t4_calibration_cache_smoke(**kwargs)
    _assert_no_usable_stage_pass(output, stages.T4_MARKER)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({}, "index binding"),
        (
            {
                "AUTODL_PHYSICAL_GPU_INDEX": "0",
                "AUTODL_PHYSICAL_GPU_UUID": "GPU-right",
                "CUDA_VISIBLE_DEVICES": "1",
            },
            "index binding",
        ),
        (
            {
                "AUTODL_PHYSICAL_GPU_INDEX": "1",
                "AUTODL_PHYSICAL_GPU_UUID": "GPU-wrong",
                "CUDA_VISIBLE_DEVICES": "1",
            },
            "UUID differs",
        ),
        (
            {
                "AUTODL_PHYSICAL_GPU_INDEX": "1",
                "AUTODL_PHYSICAL_GPU_UUID": "GPU-right",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            "visibility",
        ),
    ],
)
def test_t4_rejects_unbound_direct_child_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    message: str,
) -> None:
    for key in (
        "AUTODL_PHYSICAL_GPU_INDEX",
        "AUTODL_PHYSICAL_GPU_UUID",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(stages.TasteGNNStageError, match=message):
        stages.run_t4_calibration_cache_smoke(
            checkpoint_dir="/never-opened",
            t3_gate_path="/never-opened/gate.json",
            graph_cache_root="/never-opened",
            artifact_root="/never-opened",
            output_dir="/never-created",
            downstream_policy_path=DOWNSTREAM_POLICY,
            base_policy_path=BASE_POLICY,
            gpu_uuid="GPU-right",
        )


def test_t4_bounded_smoke_is_exactly_16_multiclass_and_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stages,
        "_real_connected_deletions",
        lambda _smiles, *, parent_id, maximum: [
            ("C", _outcome(int(parent_id.split("-")[1])))
            for _index in range(maximum)
        ],
    )
    monkeypatch.setattr(
        stages,
        "_graph_from_smiles",
        lambda _featurizer, _smiles, molecule_id: MolecularGraphData(
            x=((0,),),
            edge_index=((), ()),
            edge_attr=(),
            y=-1,
            molecule_id=molecule_id,
            smiles="C",
            split="oracle_smoke",
            graph_sha256=hashlib.sha256(molecule_id.encode()).hexdigest(),
        ),
    )
    result = stages.run_bounded_oracle_smoke(
        dataset=_FakeDataset(),
        oracle=_FakeOracle(),
        feature_schema=default_molecular_feature_schema(),
        batch_size=8,
        source_count=16,
        max_deletions_per_parent=4,
    )
    transitions = result["destination_distribution"]["overall"]["transitions"]
    assert result["status"] == "PASS"
    assert result["selected_count"] == 16
    assert result["all_selected_have_four_connected_deletions"] is True
    assert result["parent_deletion_counts_by_position"] == [4] * 16
    assert transitions["1->0"]["count"] > 0
    assert transitions["1->2"]["count"] > 0
    assert result["strict_flip_to_bitter_observed"] is True
    assert result["strict_flip_to_tasteless_observed"] is True
    assert result["checkpoint_load_count"] == 1
    assert result["per_example_predictions_written"] is False
    assert result["smiles_written"] is False
    serialized = json.dumps(result, sort_keys=True)
    assert "parent-" not in serialized
    assert "residual_smiles" not in serialized


def test_t4_bounds_reject_bool_float_and_changed_counts() -> None:
    for source_count in (True, 16.0, 15):
        with pytest.raises(stages.TasteGNNStageError, match="exactly 16"):
            stages.run_bounded_oracle_smoke(
                dataset=_FakeDataset(),
                oracle=_FakeOracle(),
                feature_schema=default_molecular_feature_schema(),
                batch_size=8,
                source_count=source_count,  # type: ignore[arg-type]
                max_deletions_per_parent=4,
            )


def test_t4_rejects_cohort_without_four_deletions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stages,
        "_real_connected_deletions",
        lambda _smiles, *, parent_id, maximum: [
            ("C", _outcome(int(parent_id.split("-")[1])))
        ],
    )
    with pytest.raises(stages.TasteGNNStageError, match="requires 16 eligible"):
        stages.run_bounded_oracle_smoke(
            dataset=_FakeDataset(),
            oracle=_FakeOracle(),
            feature_schema=default_molecular_feature_schema(),
            batch_size=8,
            source_count=16,
            max_deletions_per_parent=4,
        )


def test_t4_rejects_missing_multiclass_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stages,
        "_real_connected_deletions",
        lambda _smiles, *, parent_id, maximum: [
            ("C", _outcome(int(parent_id.split("-")[1])))
            for _index in range(maximum)
        ],
    )
    original = _FakeOracle._record

    def bitter_only(graph: MolecularGraphData) -> dict[str, Any]:
        result = original(graph)
        if graph.molecule_id.startswith("t4-residual-"):
            result = {
                **result,
                "predicted_label": 0,
                "probabilities": [0.8, 0.1, 0.1],
            }
        return result

    monkeypatch.setattr(_FakeOracle, "_record", staticmethod(bitter_only))
    with pytest.raises(stages.TasteGNNStageError, match="both Bitter and Tasteless"):
        stages.run_bounded_oracle_smoke(
            dataset=_FakeDataset(),
            oracle=_FakeOracle(),
            feature_schema=default_molecular_feature_schema(),
            batch_size=8,
            source_count=16,
            max_deletions_per_parent=4,
        )


def test_cli_wrappers_preserve_gpu1_cache_only_and_no_hpc_execution() -> None:
    core = (PROJECT_ROOT / "src/eval/tastemolnet_gnn_stages.py").read_text(
        encoding="utf-8"
    )
    t4_wrapper = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_gnn_oracle_smoke.sh"
    ).read_text(encoding="utf-8")
    t3_wrapper = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_gnn_calibration_adoption.sh"
    ).read_text(encoding="utf-8")
    slurm = (PROJECT_ROOT / "scripts/slurm/tastemolnet_gnn_stage.sh").read_text(
        encoding="utf-8"
    )
    assert "fit_temperature_scaling" not in core
    assert "autodl_new_output_dir tastemolnet gine calibrated" in t3_wrapper
    assert '--artifact-root "$AUTODL_ARTIFACT_ROOT"' in t3_wrapper
    assert '--graph-cache-root "$GRAPH_CACHE_ROOT"' in t3_wrapper
    assert "TASTEMOLNET_GRAPH_CACHE_ROOT" in t3_wrapper
    assert "--physical-gpu-index" not in t3_wrapper
    assert "--max-gpus 4" in t3_wrapper
    assert "--gpu-hard-limit 4" in t3_wrapper
    assert "calibration.pt" in t4_wrapper
    assert '--artifact-root "$AUTODL_ARTIFACT_ROOT"' in t4_wrapper
    assert "--gpu-index 1" in t4_wrapper
    assert "--physical-gpu-index 1" in t4_wrapper
    assert "--gpu-lock-mode exclusive" in t4_wrapper
    assert "--max-gpus 4" in t4_wrapper
    assert "--gpu-hard-limit 4" in t4_wrapper
    assert "predictions.csv" not in t4_wrapper
    assert "test.pt" not in t4_wrapper
    assert "[TASTE_MULTICLASS_ORACLE_PASS]" in t4_wrapper
    assert "TASTE_GINE_ORACLE_SMOKE_PASS" not in t4_wrapper
    assert "#SBATCH --partition=A800" in slurm
    assert "#SBATCH --gres=gpu:a800:1" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "--artifact-root /absolute/artifact-root" in slurm
    assert "exit 64" in slurm
    assert "HPC remains forbidden" in slurm
