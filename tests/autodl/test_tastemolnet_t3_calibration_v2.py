from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import tastemolnet_t3_calibration_v2 as t3
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.tastemolnet_t2_adoption_v2 import (
    _read_validation_predictions,
    _stable_sha256,
)
from src.utils.terminal_publisher_v2 import seal_worker_staging


def _validation_bytes(*, split: str = "validation") -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "molecule_id",
        "smiles",
        "split",
        "label",
        "predicted_label",
        "logits",
        "probabilities",
        "source_graph_hash",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    logits = ([4.0, 0.0, -1.0], [-1.0, 4.0, 0.0], [0.0, -1.0, 4.0])
    for index, (label, values) in enumerate(zip((0, 1, 2), logits, strict=True)):
        array = np.asarray([values], dtype=np.float64)
        probabilities = np.exp(array - array.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        writer.writerow(
            {
                "molecule_id": f"taste-{index}",
                "smiles": ["CC", "CN", "CO"][index],
                "split": split,
                "label": label,
                "predicted_label": label,
                "logits": json.dumps(values),
                "probabilities": json.dumps(probabilities[0].tolist()),
                "source_graph_hash": str(index + 1) * 64,
            }
        )
    return output.getvalue().encode("utf-8")


def test_fresh_temperature_records_validation_only_provenance() -> None:
    payload, evidence = t3.fit_fresh_temperature(
        _validation_bytes(),
        attempt_id="a" * 36,
        generation_token="b" * 36,
        receipt_id="receipt",
        receipt_gate_sha256="1" * 64,
        source_model_sha256="2" * 64,
        source_predictions_sha256="3" * 64,
        max_iter=10,
        fitted_at="2026-08-28T00:00:00Z",
    )
    assert payload["fit_generation"] == "FRESH_T3_REFIT"
    assert payload["temperature_refit_performed"] is True
    assert payload["selection_split"] == "validation"
    assert payload["calibration_payload_loaded"] is False
    assert payload["test_payload_loaded"] is False
    assert payload["argmax_invariant"] is True
    assert float(payload["temperature"]) > 0.0
    assert payload["validation_row_ids_sha256"] == evidence["validation_row_ids_sha256"]


def test_fresh_temperature_rejects_calibration_rows() -> None:
    with pytest.raises(Exception, match="another split"):
        t3.fit_fresh_temperature(
            _validation_bytes(split="calibration"),
            attempt_id="a" * 36,
            generation_token="b" * 36,
            receipt_id="receipt",
            receipt_gate_sha256="1" * 64,
            source_model_sha256="2" * 64,
            source_predictions_sha256="3" * 64,
            max_iter=10,
        )


def test_t3_worker_has_no_terminal_marker() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = (
        root / "scripts/autodl/tastemolnet_t3_calibration_worker_v2.py"
    ).read_text(encoding="utf-8")
    verifier = (
        root / "scripts/autodl/tastemolnet_t3_calibration_verifier_v2.py"
    ).read_text(encoding="utf-8")
    assert t3.PASS_MARKER not in worker
    assert "print(PASS_MARKER" in verifier


def test_candidate_inventory_adds_only_t3_evidence() -> None:
    assert t3.CANDIDATE_CHECKPOINT_FILES == t3.SOURCE_CHECKPOINT_FILES | {
        t3.CANDIDATE_NAME
    }
    assert t3.MODIFIED_SOURCE_FILES == {
        "model_card.json",
        "oracle_manifest.json",
        "temperature_scaling.json",
        "sha256sums.txt",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_source_bundle(root: Path) -> tuple[Path, dict[str, str], str]:
    root.mkdir(parents=True)
    predictions = _validation_bytes()
    row_hash = _read_validation_predictions(predictions)["row_ids_sha256"]
    model = b"frozen-three-class-gine\n"
    last = b"historical-last-checkpoint\n"
    model_hash = hashlib.sha256(model).hexdigest()
    last_hash = hashlib.sha256(last).hexdigest()
    policy_hash = "4" * 64
    receipt_hash = "5" * 64
    cache_hash = "6" * 64
    test_hash = "7" * 64
    model_card = {
        "dataset": "tastemolnet",
        "profile": "full",
        "checkpoint_id": model_hash,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
        "backbone": "gine",
        "feature_schema_sha256": "8" * 64,
        "data_use_policy_file_sha256": policy_hash,
        "data_use_policy_canonical_sha256": policy_hash,
        "data_use_policy_receipt_sha256": receipt_hash,
        "graph_cache_manifest_sha256": cache_hash,
    }
    policy = {
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
            "policy_file_sha256": policy_hash,
            "policy_canonical_sha256": policy_hash,
        },
        "policy_receipt": {"sha256": receipt_hash},
    }
    cache = {
        "schema_version": "tastemolnet_graph_cache_usage_v1",
        "dataset": "tastemolnet",
        "mode": "read_only_existing_cache",
        "graph_cache_used": True,
        "loaded_splits": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
        "graph_cache_rebuilt": False,
        "data_reprepared": False,
        "graph_cache_manifest_sha256": cache_hash,
    }
    oracle = {
        "schema_version": "tastemolnet_three_class_gine_oracle_manifest_v1",
        "dataset": "tastemolnet",
        "status": "PASS",
        "checkpoint_id": model_hash,
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
        "temperature_scaling": {"temperature": 1.0},
    }
    last_document = {
        "schema_version": "tastemolnet_last_training_checkpoint_v1",
        "checkpoint_file": "last.pt",
        "same_bytes_as_latest_epoch_checkpoint": True,
        "completed_epoch": 1,
        "checkpoint_sha256": last_hash,
        "source_checkpoint_sha256": last_hash,
    }
    reload_document = {
        "schema_version": "tastemolnet_gine_checkpoint_reload_v1",
        "status": "PASS",
        "checkpoint_reload_pass": True,
        "batch_single_probability_equivalence": True,
        "all_probabilities_finite": True,
        "num_classes": 3,
        "source_label": 1,
        "checkpoint_id": model_hash,
        "last_checkpoint": last_document,
    }
    split = {
        "files": {"test": {"path": "/frozen/test.csv", "sha256": test_hash}},
    }
    documents = {
        "config.yaml": {},
        "model_card.json": model_card,
        "feature_schema.json": {"schema_sha256": "8" * 64},
        "label_map.json": {"0": "Bitter", "1": "Sweet", "2": "Tasteless"},
        "split_manifest.json": split,
        "training_metrics.json": {},
        "test_evaluation_status.json": {
            "status": "NOT_EVALUATED",
            "test_loaded": False,
            "reason": "held out",
            "path": "/frozen/test.csv",
            "sha256": test_hash,
        },
        "temperature_scaling.json": {
            "schema_version": "temperature_scaling_v1",
            "status": "fit",
            "selection_split": "validation",
            "test_used_for_fit": False,
            "argmax_invariant": True,
            "num_classes": 3,
            "num_examples": 3,
            "temperature": 1.0,
        },
        "environment.json": {},
        "git_state.json": {},
        "data_use_policy_binding.json": policy,
        "graph_cache_usage.json": cache,
        "oracle_manifest.json": oracle,
        "last_checkpoint.json": last_document,
        "checkpoint_reload.json": reload_document,
    }
    (root / "model.pt").write_bytes(model)
    (root / "last.pt").write_bytes(last)
    (root / "validation_predictions.csv").write_bytes(predictions)
    for name, value in documents.items():
        _write_json(root / name, value)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
    }
    (root / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    hashes["sha256sums.txt"] = hashlib.sha256(
        (root / "sha256sums.txt").read_bytes()
    ).hexdigest()
    return root, hashes, row_hash


def _make_t2_receipt(
    root: Path,
    *,
    bundle: Path,
    hashes: dict[str, str],
    row_hash: str,
) -> Path:
    root.mkdir(parents=True)
    receipt_id = root.name
    source = {
        "receipt_id": receipt_id,
        "artifact_root": str(bundle),
        "artifact_hashes": hashes,
        "old_failure_superseded_for_scientific_artifact": True,
        "old_process_evidence_not_rewritten": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "validation_row_ids_sha256": row_hash,
    }
    source["source_evidence_sha256"] = _stable_sha256(source)
    gate = {
        "status": "PASS",
        "state": "ADOPTED_SCIENTIFIC_PASS",
        "stage": "T2_GINE",
        "receipt_id": receipt_id,
        "marker": "[TASTE_T2_GINE_ADOPTION_PASS]",
        "artifact_root": str(bundle),
        "model_sha256": hashes["model.pt"],
        "source_evidence_sha256": source["source_evidence_sha256"],
    }
    documents = {
        "artifact_hashes.json": {
            "artifact_root": str(bundle),
            "artifact_hashes": hashes,
        },
        "gate.json": gate,
        "input_hashes.json": {},
        "source_evidence.json": source,
        "verification.json": {
            "verification_result": "PASS",
            "source_evidence_sha256": source["source_evidence_sha256"],
        },
    }
    for name, value in documents.items():
        _write_json(root / name, value)
    (root / "sha256s.txt").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in sorted(documents)
        ),
        encoding="utf-8",
    )
    (root / "PASS").write_text("[TASTE_T2_GINE_ADOPTION_PASS]\n", encoding="utf-8")
    return root


def test_managed_t3_worker_and_independent_verifier_publish(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    seed_root = tmp_path / "outputs/gnn_oracles/tastemolnet/gine/seed7"
    bundle, hashes, row_hash = _make_source_bundle(seed_root / "full-source")
    receipt = _make_t2_receipt(
        tmp_path / "control/adoptions/T2_GINE/00000000-0000-4000-8000-000000000001",
        bundle=bundle,
        hashes=hashes,
        row_hash=row_hash,
    )
    stage_root = tmp_path / "control/T3_CALIBRATION"
    stage_root.mkdir(parents=True)
    controller_id = "taste-main-v2-test"
    git_commit = "a" * 40
    config_hash = "b" * 64
    input_hashes = {
        "t2_receipt_gate": hashlib.sha256((receipt / "gate.json").read_bytes()).hexdigest(),
        "t2_source_evidence": json.loads(
            (receipt / "source_evidence.json").read_text(encoding="utf-8")
        )["source_evidence_sha256"],
        "t2_source_sha256s": hashes["sha256sums.txt"],
    }
    with create_managed_attempt(
        stage_root=stage_root,
        controller_id=controller_id,
        task_id=t3.TASK_ID,
        git_commit=git_commit,
        config_hash=config_hash,
        input_hashes=input_hashes,
        attempt_id="00000000-0000-4000-8000-000000000002",
        boot_id="test-boot",
    ) as attempt:
        with create_worker_staging(
            attempt, staging_id="00000000-0000-4000-8000-000000000003"
        ) as staging:
            t3.build_t3_candidate(
                t2_receipt_root=receipt,
                source_bundle_root=bundle,
                artifact_root=staging.artifact_root,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                max_iter=10,
            )
            raw = write_worker_raw_evidence(
                staging,
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "process_lineage": {
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            raw.close()
            exited = write_worker_exit(
                staging,
                {
                    "exit_code": 0,
                    "worker_closed_artifact_writers": True,
                    "process_audit": {
                        "state": "EXITED",
                        "controller_id": controller_id,
                        "attempt_id": attempt.attempt_id,
                    },
                },
            )
            exited.close()
            sealed = seal_worker_staging(staging)
        final = seed_root / "calibrated-test"
        publication, verification = t3.verify_and_publish_t3(
            sealed_path=sealed.staging_path,
            final_path=final,
            t2_receipt_root=receipt,
            source_bundle_root=bundle,
            expected_attempt_id=sealed.attempt_id,
            expected_generation_token=sealed.generation_token,
            expected_controller_id=controller_id,
            expected_git_commit=git_commit,
            expected_config_hash=config_hash,
            max_iter=10,
        )
    assert publication.final_path == final
    assert verification["marker"] == t3.PASS_MARKER
    assert verification["temperature_refit_performed"] is True
    assert verification["model_sha256"] == hashes["model.pt"]
    assert load_verified_gate(final)["status"] == "PASS"
