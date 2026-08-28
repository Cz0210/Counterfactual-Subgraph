from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval import bace_ours_freeze_adoption as adoption
from src.eval.four_by_four_registry import CandidateAudit, CellStatus


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _policy(tmp_path: Path) -> adoption.AdoptionPolicy:
    source = tmp_path / "source"
    guard = tmp_path / "writer-guard"
    source.mkdir()
    guard.mkdir()
    files: dict[str, bytes] = {
        "PASS": b"PASS\n",
        "oracle_manifest.json": json.dumps(
            {
                "feature_schema_sha256": "f" * 64,
                "temperature_scaling_sha256": "t" * 64,
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "summary.json": json.dumps(
            {
                "num_classes": 2,
                "source_label": 1,
                "rf_oracle_used": False,
                "selection_frozen_before_test": True,
                "selector_fitted_on_calibration": True,
                "test_loaded_only_after_freeze": True,
                "test_used_for_selection": False,
                "threshold_fitted_on_test": False,
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "evaluation_manifest.json": json.dumps(
            {
                "candidate_order_changed": False,
                "scientific_metrics_recomputed": False,
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "final_artifact_audit.json": json.dumps(
            {
                "final_artifact_audit_passed": True,
                "hash_closure_complete": True,
                "no_numeric_imputation": True,
            },
            sort_keys=True,
        ).encode()
        + b"\n",
    }
    for name, payload in files.items():
        (source / name).write_bytes(payload)
    expected = {
        "cf_mode": "strict_flip",
        "classifier_family": "gine",
        "dataset_hash": "d" * 64,
        "feature_schema_sha256": "f" * 64,
        "k_max": 20,
        "method": "Ours",
        "molclr_checkpoint_hash": "m" * 64,
        "num_classes": 2,
        "oracle_backend": "gnn",
        "oracle_checkpoint": "/frozen/bace/oracle",
        "oracle_hash": "o" * 64,
        "rf_oracle_used": False,
        "source_label": 1,
        "split_hash": "s" * 64,
        "table2_k": 10,
        "temperature_scaling_sha256": "t" * 64,
        "threshold_config_hash": "h" * 64,
    }
    policy_path = tmp_path / "policy.json"
    _write_json(
        policy_path,
        {
            "schema_version": adoption.POLICY_SCHEMA,
            "dataset": "BACE",
            "method": "Ours",
            "source_root": str(source),
            "writer_guard_root": str(guard),
            "expected_identity": expected,
            "source_files": {name: _sha(payload) for name, payload in files.items()},
        },
    )
    return adoption.load_policy(policy_path)


def _candidate(policy: adoption.AdoptionPolicy) -> CandidateAudit:
    expected = policy.expected_identity
    row = {
        "cf_mode": expected["cf_mode"],
        "classifier_family": expected["classifier_family"],
        "dataset_hash": expected["dataset_hash"],
        "k_max": expected["k_max"],
        "method": expected["method"],
        "molclr_checkpoint_hash": expected["molclr_checkpoint_hash"],
        "oracle_backend": expected["oracle_backend"],
        "oracle_checkpoint": expected["oracle_checkpoint"],
        "oracle_hash": expected["oracle_hash"],
        "split_hash": expected["split_hash"],
        "table2_k": expected["table2_k"],
        "threshold_config_hash": expected["threshold_config_hash"],
    }
    return CandidateAudit(
        root=policy.source_root,
        dataset="BACE",
        method="Ours",
        status=CellStatus.FROZEN_PASS,
        reason_codes=[],
        row=row,
        artifact_hashes={},
    )


def _writer_audit(*args: object, **kwargs: object) -> dict[str, object]:
    return {
        "procfs_verified": True,
        "scanned_process_count": 7,
        "writable_fd_count": 0,
        "writers": [],
    }


def test_receipt_only_adoption_is_closed_and_reopenable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    output = matrix / "adoptions/bace_ours_frozen_test"

    result = adoption.adopt_bace_ours_frozen_cell(
        matrix_root=matrix,
        output_root=output,
        policy=policy,
        require_clean_git=False,
    )

    assert result["marker"] == adoption.PASS_MARKER
    assert sorted(path.name for path in output.iterdir()) == [
        "PASS",
        "adoption_manifest.json",
        "verification.json",
    ]
    assert (output / "PASS").read_text(encoding="ascii") == (
        adoption.PASS_MARKER + "\n"
    )
    assert adoption.validate_adoption_receipt(output, policy=policy)["status"] == "PASS"


def test_adoption_rejects_drift_and_non_frozen_registry_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    candidate = _candidate(policy)
    candidate.status = CellStatus.INCOMPLETE
    candidate.reason_codes = ["TEST_BLOCK"]
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: candidate
    )
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="registry gate"):
        adoption.validate_source_candidate(policy)

    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )
    (policy.source_root / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="identity changed"):
        adoption.validate_source_candidate(policy)


def test_adoption_rejects_wrong_destination_and_receipt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="fresh"):
        adoption.adopt_bace_ours_frozen_cell(
            matrix_root=matrix,
            output_root=matrix / "adoptions/not_bace",
            policy=policy,
            require_clean_git=False,
        )

    output = matrix / "adoptions/bace_ours_frozen_tamper"
    adoption.adopt_bace_ours_frozen_cell(
        matrix_root=matrix,
        output_root=output,
        policy=policy,
        require_clean_git=False,
    )
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    verification["scientific_recomputation_performed"] = True
    _write_json(output / "verification.json", verification)
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="closure changed"):
        adoption.validate_adoption_receipt(output, policy=policy)


def test_policy_rejects_changes_to_fixed_scientific_identity(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    payload = json.loads(policy.path.read_text(encoding="utf-8"))
    payload["expected_identity"]["k_max"] = 19
    _write_json(policy.path, payload)
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="fixed BACE"):
        adoption.load_policy(policy.path)
