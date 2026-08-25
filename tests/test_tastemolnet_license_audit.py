from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_tastemolnet_license import (
    BLOCKED,
    UPSTREAM_COMMIT,
    LicenseAuditError,
    audit_license,
)


def test_audit_completion_mode_does_not_unlock_blocked_gate(
    tmp_path: Path, capsys
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "tastemolnet",
                "upstream_commit": UPSTREAM_COMMIT,
                "license_id": None,
                "license_status": "LICENSE_REVIEW_REQUIRED",
                "raw_data_commit_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    from scripts import audit_tastemolnet_license as module

    assert (
        module.main(
            [
                "--prepared-root",
                str(prepared),
                "--output-dir",
                str(output),
                "--audit-completion-mode",
            ]
        )
        == 0
    )
    gate = json.loads((output / "taste_license_gate.json").read_text())
    assert gate["status"] == BLOCKED
    assert gate["passed"] is False
    assert gate["heavy_route_authorized"] is False
    stdout = capsys.readouterr().out
    assert "[TASTE_LICENSE_BLOCKED]" in stdout
    assert "[TASTE_LICENSE_AUDIT_COMPLETE] status=BLOCKED_LICENSE_REVIEW" in stdout


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
    assert gate["passed"] is False
    assert gate["heavy_route_authorized"] is False
    assert (tmp_path / "audit" / BLOCKED).is_file()
    assert not (tmp_path / "audit" / "PASS").exists()


def test_nonempty_user_terms_never_create_legacy_license_pass(tmp_path: Path) -> None:
    approval = tmp_path / "approval.txt"
    approval.write_text("Written permission covers research use of the exact CSV.\n")
    gate = audit_license(
        prepared_root=_prepared(tmp_path),
        output_dir=tmp_path / "audit",
        approval_file=approval,
    )
    assert gate["status"] == BLOCKED
    assert gate["passed"] is False
    assert gate["heavy_route_authorized"] is False
    assert gate["license_basis"] is None
    assert gate["approval_file"] == str(approval.resolve())
    assert gate["approval_evidence"]["sha256"]
    assert gate["legacy_license_pass_disabled"] is True
    assert not (tmp_path / "audit" / "PASS").exists()


def test_reviewed_prepared_terms_still_do_not_activate_legacy_gate(tmp_path: Path) -> None:
    gate = audit_license(
        prepared_root=_prepared(
            tmp_path,
            license_id="CC-BY-4.0",
            license_status="REVIEWED:CC-BY-4.0",
            raw_data_commit_allowed=True,
        ),
        output_dir=tmp_path / "audit",
    )
    assert gate["status"] == BLOCKED
    assert gate["passed"] is False
    assert gate["heavy_route_authorized"] is False
    assert gate["license_basis"] is None
    assert gate["legacy_license_pass_disabled"] is True


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
