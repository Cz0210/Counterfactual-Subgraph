#!/usr/bin/env python3
"""Freeze a fail-closed TasteMolNet data-license decision.

This command never downloads data and never infers permission merely from a
public repository or an open-access paper.  A PASS requires either explicit
license evidence already bound by the prepared-data provenance or a
user-supplied approval/terms file.  Otherwise it records the available
evidence and emits ``BLOCKED_LICENSE_REVIEW``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "tastemolnet_license_audit_v1"
BLOCKED = "BLOCKED_LICENSE_REVIEW"
PASS = "PASS"
UPSTREAM_REPOSITORY = "https://github.com/MujeebOnawole/Taste_Prediction_RGCN"
UPSTREAM_COMMIT = "16af8ead8a17b6bd3941d9eb5879c5be75c14114"
UPSTREAM_DATA_FILE = "processed_data/taste_scaffold_split.csv"


class LicenseAuditError(RuntimeError):
    """The license decision cannot be made without weakening provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LicenseAuditError(f"Expected one JSON object: {path}")
    return payload


def _explicit_prepared_license(provenance: dict[str, Any]) -> tuple[bool, str | None]:
    license_id = str(provenance.get("license_id") or "").strip()
    status = str(provenance.get("license_status") or "").strip()
    allowed = provenance.get("raw_data_commit_allowed") is True
    reviewed = bool(license_id) and status == f"REVIEWED:{license_id}" and allowed
    return reviewed, license_id or None


def audit_license(
    *,
    prepared_root: str | Path,
    output_dir: str | Path,
    approval_file: str | Path | None = None,
    upstream_checkout: str | Path | None = None,
) -> dict[str, Any]:
    prepared = Path(prepared_root).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"License audit output must be fresh: {output}")
    provenance_path = prepared / "provenance_manifest.json"
    if not provenance_path.is_file():
        raise LicenseAuditError(f"Prepared provenance is missing: {provenance_path}")
    provenance = _read_object(provenance_path)
    dataset = str(provenance.get("dataset") or "").lower()
    if dataset != "tastemolnet":
        raise LicenseAuditError(f"Unexpected prepared dataset: {dataset!r}")
    commit = str(provenance.get("upstream_commit") or "")
    if commit != UPSTREAM_COMMIT:
        raise LicenseAuditError(
            f"TasteMolNet upstream commit drift: expected {UPSTREAM_COMMIT}, got {commit}"
        )

    reviewed, license_id = _explicit_prepared_license(provenance)
    approval_identity: dict[str, Any] | None = None
    if approval_file is not None:
        approval = Path(approval_file).expanduser().resolve(strict=True)
        if not approval.is_file() or approval.is_symlink():
            raise LicenseAuditError("Approval evidence must be one physical file.")
        text = approval.read_text(encoding="utf-8").strip()
        if not text:
            raise LicenseAuditError("Approval evidence is empty.")
        approval_identity = {
            "path": str(approval),
            "sha256": _sha256(approval),
            "size": approval.stat().st_size,
            "supplied_by_user": True,
        }

    checkout_evidence: dict[str, Any] = {
        "inspected": False,
        "license_files": [],
        "readme_mentions_data_license": False,
    }
    if upstream_checkout is not None:
        checkout = Path(upstream_checkout).expanduser().resolve(strict=True)
        license_files = [
            child.name
            for child in checkout.iterdir()
            if child.is_file()
            and child.name.lower().split(".", 1)[0]
            in {"license", "copying", "notice"}
        ]
        readme = checkout / "README.md"
        readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        lowered = readme_text.lower()
        checkout_evidence = {
            "inspected": True,
            "path": str(checkout),
            "license_files": sorted(license_files),
            "readme_present": readme.is_file(),
            "readme_mentions_data_license": any(
                token in lowered
                for token in ("dataset license", "data license", "licensed under")
            ),
        }

    passed_by = None
    if approval_identity is not None:
        passed_by = "user_supplied_approval_or_terms"
    elif reviewed:
        passed_by = "prepared_provenance_explicit_license"
    status = PASS if passed_by else BLOCKED
    now = datetime.now(timezone.utc).isoformat()
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "dataset": "tastemolnet",
        "prepared_root": str(prepared),
        "prepared_provenance": {
            "path": str(provenance_path),
            "sha256": _sha256(provenance_path),
            "source_csv_sha256": provenance.get("source_csv_sha256"),
            "upstream_commit": commit,
            "license_id": provenance.get("license_id"),
            "license_status": provenance.get("license_status"),
            "raw_data_commit_allowed": provenance.get("raw_data_commit_allowed"),
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "data_file": UPSTREAM_DATA_FILE,
            "public_access_is_not_permission": True,
            "repository_license_not_established": True,
            "underlying_bst_rights_not_established": True,
        },
        "upstream_checkout": checkout_evidence,
        "approval_evidence": approval_identity,
        "decision_basis": passed_by or "no_explicit_data_reuse_license_or_approval",
        "legal_advice": False,
    }
    gate = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "dataset": "tastemolnet",
        "status": status,
        "passed": status == PASS,
        "license_id": license_id,
        "license_basis": passed_by,
        "reuse_basis": passed_by,
        "approval_file": (
            approval_identity.get("path") if approval_identity is not None else None
        ),
        "approval_evidence": approval_identity,
        "heavy_route_authorized": status == PASS,
        "run_tastemolnet": status == PASS,
        "blocked_reason": None if status == PASS else BLOCKED,
        "required_for_unblock": (
            None
            if status == PASS
            else "Explicit terms covering this CSV/data compilation or a user-supplied approval file."
        ),
        "evidence_file": "taste_license_evidence.json",
    }
    markdown = "\n".join(
        [
            "# TasteMolNet license audit",
            "",
            f"- Status: `{status}`",
            f"- Upstream commit: `{UPSTREAM_COMMIT}`",
            f"- Prepared provenance: `{provenance_path}`",
            f"- Decision basis: `{evidence['decision_basis']}`",
            "- Public availability of a repository, paper, or CSV was not treated as a data license.",
            "- No TasteMolNet full GNN, PPO, generation, verification, selector, or final evaluation is authorized while blocked.",
            "",
            "To unblock, provide `TASTEMOLNET_LICENSE_APPROVAL_FILE` pointing to explicit terms or written approval that covers research reuse of the exact data file.",
            "",
        ]
    )
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "taste_license_evidence.json", evidence)
    _atomic_json(output / "taste_license_gate.json", gate)
    _atomic_text(output / "taste_license_audit.md", markdown)
    _atomic_text(output / ("PASS" if status == PASS else BLOCKED), status + "\n")
    return gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--approval-file", default=os.environ.get("TASTEMOLNET_LICENSE_APPROVAL_FILE")
    )
    parser.add_argument("--upstream-checkout")
    parser.add_argument(
        "--audit-completion-mode",
        action="store_true",
        help=(
            "Return success after publishing a complete BLOCKED audit. This does "
            "not change taste_license_gate.json or authorize heavy work."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate = audit_license(
        prepared_root=args.prepared_root,
        output_dir=args.output_dir,
        approval_file=args.approval_file,
        upstream_checkout=args.upstream_checkout,
    )
    print(json.dumps(gate, sort_keys=True), flush=True)
    marker = "TASTE_LICENSE_PASS" if gate["status"] == PASS else "TASTE_LICENSE_BLOCKED"
    print(f"[{marker}]", flush=True)
    if args.audit_completion_mode:
        print(f"[TASTE_LICENSE_AUDIT_COMPLETE] status={gate['status']}", flush=True)
        return 0
    return 0 if gate["status"] == PASS else 65


if __name__ == "__main__":
    raise SystemExit(main())
