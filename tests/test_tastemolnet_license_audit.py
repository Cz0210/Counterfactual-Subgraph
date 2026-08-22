from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_tastemolnet_license import (
    BLOCKED,
    PASS,
    UPSTREAM_COMMIT,
    LicenseAuditError,
    audit_license,
)


def _prepared(tmp_path: Path, **overrides: object) -> Path:
    root = tmp_path / "prepared"
    root.mkdir()
    payload = {
        "dataset": "tastemolnet",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_csv_sha256": "a" * 64,
        "license_id": None,
        "license_status": "LICENSE_REVIEW_REQUIRED",
        "raw_data_commit_allowed": False,
    }
    payload.update(overrides)
    (root / "provenance_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return root


def test_public_unlicensed_data_stays_blocked(tmp_path: Path) -> None:
    gate = audit_license(
        prepared_root=_prepared(tmp_path), output_dir=tmp_path / "audit"
    )
    assert gate["status"] == BLOCKED
    assert gate["heavy_route_authorized"] is False
    assert (tmp_path / "audit" / BLOCKED).is_file()
    assert not (tmp_path / "audit" / "PASS").exists()


def test_nonempty_user_approval_unlocks_exact_prepared_data(tmp_path: Path) -> None:
    approval = tmp_path / "approval.txt"
    approval.write_text("Written permission covers research use of the exact CSV.\n")
    gate = audit_license(
        prepared_root=_prepared(tmp_path),
        output_dir=tmp_path / "audit",
        approval_file=approval,
    )
    assert gate["status"] == PASS
    assert gate["heavy_route_authorized"] is True
    assert gate["approval_evidence"]["sha256"]


def test_reviewed_prepared_license_requires_consistent_fields(tmp_path: Path) -> None:
    gate = audit_license(
        prepared_root=_prepared(
            tmp_path,
            license_id="CC-BY-4.0",
            license_status="REVIEWED:CC-BY-4.0",
            raw_data_commit_allowed=True,
        ),
        output_dir=tmp_path / "audit",
    )
    assert gate["status"] == PASS


def test_commit_drift_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(LicenseAuditError, match="commit drift"):
        audit_license(
            prepared_root=_prepared(tmp_path, upstream_commit="b" * 40),
            output_dir=tmp_path / "audit",
        )


def test_output_is_fresh_only(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    output = tmp_path / "audit"
    audit_license(prepared_root=prepared, output_dir=output)
    with pytest.raises(FileExistsError, match="fresh"):
        audit_license(prepared_root=prepared, output_dir=output)
