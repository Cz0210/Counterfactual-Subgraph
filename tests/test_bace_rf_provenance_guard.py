import pytest

from scripts.audit_oracle_provenance import (
    OracleProvenanceError,
    assert_oracle_provenance,
    audit_oracle_provenance,
)


def _valid_payload(dataset: str, num_classes: int) -> dict:
    return {
        "dataset": dataset,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "num_classes": num_classes,
        "source_label": 1,
        "checkpoint": "model.pt",
    }


@pytest.mark.parametrize(("dataset", "num_classes"), [("bace", 2), ("tastemolnet", 3)])
def test_formal_gnn_provenance_passes(dataset: str, num_classes: int) -> None:
    report = assert_oracle_provenance(_valid_payload(dataset, num_classes))
    assert report["passed"] is True
    assert report["forbidden_rf_references"] == []


@pytest.mark.parametrize("dataset", ["bace", "tastemolnet"])
def test_rf_backend_or_pickle_reference_fails_closed(dataset: str) -> None:
    payload = _valid_payload(dataset, 2 if dataset == "bace" else 3)
    payload.update(
        {
            "oracle_backend": "rf",
            "classifier_type": "RandomForest",
            "rf_oracle_used": True,
            "teacher_path": "outputs/oracle/legacy_teacher.pkl",
        }
    )
    report = audit_oracle_provenance(payload)
    assert report["passed"] is False
    assert "forbidden_rf_provenance_reference" in report["errors"]
    with pytest.raises(OracleProvenanceError):
        assert_oracle_provenance(payload)


def test_missing_required_provenance_fields_is_not_a_pass() -> None:
    report = audit_oracle_provenance({"dataset": "bace", "backbone": "gine"})
    assert report["passed"] is False
    assert "oracle_backend_missing" in report["errors"]
    assert "classifier_type_missing" in report["errors"]
    assert "rf_oracle_used_missing" in report["errors"]
