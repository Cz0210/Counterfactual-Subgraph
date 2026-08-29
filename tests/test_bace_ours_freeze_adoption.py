from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.eval import bace_ours_freeze_adoption as adoption
from src.eval.four_by_four_registry import CandidateAudit, CellStatus


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _policy(
    tmp_path: Path, *, tampered_classifier_manifest: str | None = None
) -> adoption.AdoptionPolicy:
    source = tmp_path / "source"
    guard = tmp_path / "writer-guard"
    source.mkdir(parents=True)
    guard.mkdir()
    files: dict[str, bytes] = {
        "PASS": b"PASS\n",
        "oracle_manifest.json": json.dumps(
            {
                "classifier_family": (
                    "gin"
                    if tampered_classifier_manifest == "oracle_manifest.json"
                    else "gine"
                ),
                "feature_schema_sha256": "f" * 64,
                "temperature_scaling_sha256": "a" * 64,
            },
            sort_keys=True,
        ).encode()
        + b"\n",
        "summary.json": json.dumps(
            {
                "classifier_family": (
                    "gin"
                    if tampered_classifier_manifest == "summary.json"
                    else "gine"
                ),
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
        "run_manifest.json": json.dumps(
            {
                "classifier_family": (
                    "gin"
                    if tampered_classifier_manifest == "run_manifest.json"
                    else "gine"
                ),
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
                "classifier_family": (
                    "gin"
                    if tampered_classifier_manifest == "final_artifact_audit.json"
                    else "gine"
                ),
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
        "dataset_hash": "1" * 64,
        "feature_schema_sha256": "f" * 64,
        "k_max": 20,
        "method": "Ours",
        "molclr_checkpoint_hash": "b" * 64,
        "num_classes": 2,
        "oracle_backend": "gnn",
        "oracle_checkpoint": "/frozen/bace/oracle",
        "oracle_hash": "c" * 64,
        "rf_oracle_used": False,
        "source_label": 1,
        "split_hash": "d" * 64,
        "table2_k": 10,
        "temperature_scaling_sha256": "a" * 64,
        "threshold_config_hash": "e" * 64,
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


def test_realistic_frozen_registry_row_omits_classifier_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    candidate = _candidate(policy)
    assert candidate.status is CellStatus.FROZEN_PASS
    assert "classifier_family" not in candidate.row
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: candidate
    )

    evidence = adoption.validate_source_candidate(policy)

    assert evidence["registry_status"] == "FROZEN_PASS"
    assert "classifier_family" not in evidence["registry_row"]


def test_source_inventory_records_exact_physical_stat_fields(tmp_path: Path) -> None:
    policy = _policy(tmp_path)

    inventory = adoption._source_inventory(policy)

    for name, recorded in inventory.items():
        observed = (policy.source_root / name).stat(follow_symlinks=False)
        assert recorded == {
            "bytes": observed.st_size,
            "sha256": policy.source_files[name],
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mtime_ns": observed.st_mtime_ns,
            "ctime_ns": observed.st_ctime_ns,
            "link_count": observed.st_nlink,
        }


def test_receipt_rejects_source_hardlink_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    output = matrix / "adoptions/bace_ours_frozen_hardlink_drift"
    adoption.adopt_bace_ours_frozen_cell(
        matrix_root=matrix,
        output_root=output,
        policy=policy,
        require_clean_git=False,
    )
    source_file = policy.source_root / "summary.json"
    before = source_file.stat(follow_symlinks=False)

    os.link(source_file, tmp_path / "summary-hardlink.json")
    after = source_file.stat(follow_symlinks=False)

    assert after.st_nlink == before.st_nlink + 1
    assert after.st_ctime_ns >= before.st_ctime_ns
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="closure changed"):
        adoption.validate_adoption_receipt(output, policy=policy)


@pytest.mark.parametrize(
    "manifest_name",
    [
        "summary.json",
        "oracle_manifest.json",
        "run_manifest.json",
        "final_artifact_audit.json",
    ],
)
def test_direct_classifier_family_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_name: str,
) -> None:
    policy = _policy(tmp_path, tampered_classifier_manifest=manifest_name)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )

    with pytest.raises(
        adoption.BACEOursFreezeAdoptionError,
        match=rf"classifier family changed in {manifest_name}",
    ):
        adoption.validate_source_candidate(policy)


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


def test_post_publish_validation_failure_cannot_leave_terminal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(adoption, "scan_live_writers", _writer_audit)
    monkeypatch.setattr(
        adoption, "audit_explicit_candidate", lambda *args, **kwargs: _candidate(policy)
    )
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    output = matrix / "adoptions/bace_ours_frozen_post_publish_failure"

    def _reject(*args: object, **kwargs: object) -> dict[str, object]:
        raise adoption.BACEOursFreezeAdoptionError("forced post-publish rejection")

    monkeypatch.setattr(adoption, "_validate_adoption_receipt", _reject)
    with pytest.raises(
        adoption.BACEOursFreezeAdoptionError, match="forced post-publish rejection"
    ):
        adoption.adopt_bace_ours_frozen_cell(
            matrix_root=matrix,
            output_root=output,
            policy=policy,
            require_clean_git=False,
        )

    assert output.is_dir()
    assert not (output / "PASS").exists()
    assert sorted(path.name for path in output.iterdir()) == [
        "adoption_manifest.json",
        "verification.json",
    ]


def test_pass_directory_fsync_failure_removes_owned_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt"
    output.mkdir()
    expected = adoption._stat_identity(output)
    original_fsync = adoption.os.fsync
    calls = 0

    def _fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced PASS directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(adoption.os, "fsync", _fail_directory_fsync)
    with pytest.raises(OSError, match="forced PASS directory fsync failure"):
        adoption._publish_pass_last(
            output,
            expected_directory_identity=expected,
        )

    assert not (output / "PASS").exists()


def test_policy_rejects_changes_to_fixed_scientific_identity(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    payload = json.loads(policy.path.read_text(encoding="utf-8"))
    payload["expected_identity"]["k_max"] = 19
    _write_json(policy.path, payload)
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="fixed BACE"):
        adoption.load_policy(policy.path)


def test_policy_requires_absolute_roots_and_sha256_identities(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    payload = json.loads(policy.path.read_text(encoding="utf-8"))
    payload["expected_identity"]["oracle_hash"] = "not-a-sha"
    _write_json(policy.path, payload)
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="hash identity"):
        adoption.load_policy(policy.path)

    policy = _policy(tmp_path / "second")
    payload = json.loads(policy.path.read_text(encoding="utf-8"))
    payload["source_root"] = "relative/source"
    _write_json(policy.path, payload)
    with pytest.raises(adoption.BACEOursFreezeAdoptionError, match="must be absolute"):
        adoption.load_policy(policy.path)
