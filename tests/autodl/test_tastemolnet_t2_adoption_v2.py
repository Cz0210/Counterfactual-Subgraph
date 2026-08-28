from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from src.oracles.gnn_oracle import classification_metrics
from src.utils import tastemolnet_t2_adoption_v2 as adoption


def _prediction_bytes(*, collapsed: bool = False) -> bytes:
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
    logits = ([8.0, -2.0, -3.0], [-2.0, 8.0, -3.0], [-3.0, -2.0, 8.0])
    if collapsed:
        logits = ([8.0, -2.0, -3.0], [8.0, -2.0, -3.0], [8.0, -2.0, -3.0])
    for index, (label, row_logits) in enumerate(zip((0, 1, 2), logits, strict=True)):
        values = np.asarray([row_logits], dtype=np.float64)
        probabilities = adoption._softmax(values)[0].tolist()
        writer.writerow(
            {
                "molecule_id": f"taste-{index}",
                "smiles": ["CC", "CN", "CO"][index],
                "split": "val",
                "label": str(label),
                "predicted_label": str(int(np.argmax(probabilities))),
                "logits": json.dumps(list(row_logits)),
                "probabilities": json.dumps(probabilities),
                "source_graph_hash": str(index + 1) * 64,
            }
        )
    return output.getvalue().encode("utf-8")


def _temperature_payload(stored: dict[str, object], scalar: float = 2.0) -> dict[str, object]:
    logits = np.asarray(stored["logits"], dtype=np.float64)
    labels = np.asarray(stored["labels"], dtype=np.int64)
    before = adoption._softmax(logits)
    after = adoption._softmax(logits / scalar)
    before_metrics = classification_metrics(labels, before, num_classes=3)
    after_metrics = classification_metrics(labels, after, num_classes=3)
    return {
        "schema_version": "temperature_scaling_v1",
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "argmax_invariant": True,
        "num_classes": 3,
        "num_examples": len(labels),
        "temperature": scalar,
        "nll_before": float(-np.log(before[np.arange(len(labels)), labels]).mean()),
        "nll_after": float(-np.log(after[np.arange(len(labels)), labels]).mean()),
        "ece_before": float(before_metrics["ece"]),
        "ece_after": float(after_metrics["ece"]),
        "brier_before": float(before_metrics["brier_score"]),
        "brier_after": float(after_metrics["brier_score"]),
    }


def test_validation_evidence_is_three_class_and_finite() -> None:
    evidence = adoption._read_validation_predictions(_prediction_bytes())
    assert evidence["predicted_classes"] == [0, 1, 2]
    assert evidence["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert len(evidence["row_ids_sha256"]) == 64


def test_validation_evidence_rejects_single_class_collapse() -> None:
    with pytest.raises(adoption.TasteT2AdoptionError, match="fewer than 3"):
        adoption._read_validation_predictions(_prediction_bytes(collapsed=True))


def test_historical_temperature_is_authenticated_but_not_adopted_as_t3() -> None:
    stored = adoption._read_validation_predictions(_prediction_bytes())
    evidence = adoption._audit_historical_temperature(_temperature_payload(stored), stored)
    assert evidence["status"] == "PASS"
    assert evidence["fresh_T3_refit_still_required"] is True
    assert evidence["test_used_for_fit"] is False


def test_historical_temperature_rejects_test_fit() -> None:
    stored = adoption._read_validation_predictions(_prediction_bytes())
    payload = _temperature_payload(stored)
    payload["test_used_for_fit"] = True
    with pytest.raises(adoption.TasteT2AdoptionError, match="temperature evidence"):
        adoption._audit_historical_temperature(payload, stored)


def test_sha_inventory_must_close_exact_bundle() -> None:
    lines = b"".join(
        f"{'0' * 64}  {name}\n".encode("utf-8")
        for name in sorted(adoption.REQUIRED_BUNDLE_FILES - {"sha256s.txt"})
    )
    assert set(adoption._parse_sha256s(lines)) == adoption.REQUIRED_BUNDLE_FILES - {
        "sha256s.txt"
    }
    with pytest.raises(adoption.TasteT2AdoptionError, match="exact bundle"):
        adoption._parse_sha256s(lines + f"{'1' * 64}  extra.json\n".encode("utf-8"))


def test_historical_identity_drift_remains_failed(tmp_path: Path) -> None:
    controller = tmp_path / "controller"
    training = tmp_path / "training"
    controller.mkdir()
    training.mkdir()
    (controller / "controller_state.json").write_text(
        json.dumps({"phase": "FAILED", "reason": adoption.EXPECTED_FAILURE_REASON}),
        encoding="utf-8",
    )
    (controller / "controller_spec.json").write_text("{}", encoding="utf-8")
    (training / "training_complete.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    held, evidence = adoption._verify_historical_state(controller, training)
    try:
        assert evidence["old_terminal_state"] == "FAILED"
        assert evidence["old_failure_reason"] == adoption.EXPECTED_FAILURE_REASON
    finally:
        for item in held:
            item.close()


def test_historical_controller_cannot_be_rewritten_to_pass(tmp_path: Path) -> None:
    controller = tmp_path / "controller"
    training = tmp_path / "training"
    controller.mkdir()
    training.mkdir()
    (controller / "controller_state.json").write_text(
        json.dumps({"phase": "PASS", "reason": None}), encoding="utf-8"
    )
    (controller / "controller_spec.json").write_text("{}", encoding="utf-8")
    (training / "training_complete.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    with pytest.raises(adoption.TasteT2AdoptionError, match="retained identity-drift"):
        adoption._verify_historical_state(controller, training)


def test_receipt_namespace_and_marker_are_v2_exact() -> None:
    assert str(adoption.RECEIPT_NAMESPACE) == "tastemolnet-main-v2/adoptions/T2_GINE"
    assert adoption.PASS_MARKER == "[TASTE_T2_GINE_ADOPTION_PASS]"
    worker = (
        Path(__file__).resolve().parents[2] / "scripts/autodl/managed_worker_v2.py"
    ).read_text(encoding="utf-8")
    assert adoption.PASS_MARKER not in worker
