"""Read-only gate preventing ablation science before the final main table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.autodl.append_bace_gcf_matrix_authority import _verify_authority
from src.ablations.contracts import (
    ContractError,
    MAIN_ARTIFACT_KINDS,
    validate_main_artifact_receipt,
    validate_run_authorization_receipt,
)
from src.eval.fast16_matrix_authority_pointer import POINTER_SCHEMA
from src.eval.four_by_four_registry import DATASETS, METHODS, PASS_STATUSES


EXPECTED_MAIN_CELLS = 16
EXPECTED_MAIN_CELL_NAMES = tuple(
    f"{dataset}/{method}" for dataset in DATASETS for method in METHODS
)


@dataclass(frozen=True, slots=True)
class LaunchGateDecision:
    state: str
    science_launch_allowed: bool
    main_matrix_complete_cells: int
    main_matrix_total_cells: int
    final_audit_pass: bool
    figure3_pass: bool
    figure4_pass: bool
    table2_pass: bool
    explicit_run_authorization: bool
    run_requested: bool
    authority_verified: bool
    authority_root: str | None
    matrix_status_sha256: str | None
    combined_audit_sha256: str | None
    artifact_receipts_bound: bool
    authorization_receipt_sha256: str | None
    evidence_errors: tuple[str, ...]
    reasons: tuple[str, ...]
    schema_version: str = "ablation_launch_gate_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["evidence_errors"] = list(self.evidence_errors)
        return payload


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_matrix_authority_pointer(
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the exact pointer and its hash-closed matrix authority root."""

    if pointer.get("schema_version") != POINTER_SCHEMA:
        raise ContractError("matrix authority pointer schema changed")
    raw_root = str(pointer.get("latest_authority_root") or "")
    lexical = Path(raw_root).expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise ContractError("matrix authority pointer root must be absolute and physical")
    try:
        authority = _verify_authority(lexical)
    except Exception as exc:
        raise ContractError(f"matrix authority root failed independent reopen: {exc}") from exc
    passing = {status.value for status in PASS_STATUSES}
    applied = [
        f"{dataset}/{method}"
        for dataset in DATASETS
        for method in METHODS
        if str(authority["rows"][(dataset, method)].get("status") or "") in passing
    ]
    expected_pointer = {
        "schema_version": POINTER_SCHEMA,
        "latest_authority_root": str(authority["root"]),
        "latest_count": int(authority["complete"]),
        "latest_matrix_status_sha256": str(authority["matrix_sha256"]),
        "latest_combined_audit_sha256": str(authority["combined_sha256"]),
        "applied_cells": applied,
    }
    if dict(pointer) != expected_pointer:
        changed = sorted(
            key
            for key in set(pointer) | set(expected_pointer)
            if pointer.get(key) != expected_pointer.get(key)
        )
        raise ContractError(
            "matrix authority pointer/root closure changed: " + ", ".join(changed)
        )
    exact_16 = (
        int(authority["complete"]) == EXPECTED_MAIN_CELLS
        and applied == list(EXPECTED_MAIN_CELL_NAMES)
    )
    return {
        "root": str(authority["root"]),
        "matrix_status_sha256": str(authority["matrix_sha256"]),
        "combined_audit_sha256": str(authority["combined_sha256"]),
        "complete_cells": int(authority["complete"]),
        "applied_cells": applied,
        "exact_16_cells": exact_16,
        "cell_roots": {
            f"{dataset}/{method}": str(
                authority["rows"][(dataset, method)].get("standardized_output_root")
                or ""
            )
            for dataset in DATASETS
            for method in METHODS
        },
    }


def evaluate_launch_gate(
    *,
    family: str,
    matrix_authority: Mapping[str, Any],
    final_audit: Mapping[str, Any] | None,
    figure3: Mapping[str, Any] | None,
    figure4: Mapping[str, Any] | None,
    table2: Mapping[str, Any] | None,
    authorization_receipt: Mapping[str, Any] | None,
    run_requested: bool,
) -> LaunchGateDecision:
    if family not in {"llm", "gnn"}:
        raise ContractError(f"unsupported ablation family: {family}")
    evidence_errors: list[str] = []
    try:
        authority = validate_matrix_authority_pointer(matrix_authority)
        authority_verified = True
    except ContractError as exc:
        authority = {
            "root": None,
            "matrix_status_sha256": None,
            "combined_audit_sha256": None,
            "complete_cells": 0,
            "exact_16_cells": False,
        }
        authority_verified = False
        evidence_errors.append(f"MATRIX_AUTHORITY_INVALID:{exc}")
    count = int(authority["complete_cells"])
    complete = authority_verified and authority["exact_16_cells"] is True

    raw_receipts = {
        "FINAL_AUDIT": final_audit,
        "FIGURE3": figure3,
        "FIGURE4": figure4,
        "TABLE2": table2,
    }
    receipts: dict[str, dict[str, Any]] = {}
    if authority_verified:
        for kind in MAIN_ARTIFACT_KINDS:
            raw = raw_receipts[kind]
            if not isinstance(raw, Mapping):
                continue
            try:
                receipts[kind] = validate_main_artifact_receipt(
                    raw,
                    artifact_kind=kind,
                    authority=authority,
                )
            except ContractError as exc:
                evidence_errors.append(f"{kind}_RECEIPT_INVALID:{exc}")
    final_ok = "FINAL_AUDIT" in receipts
    figure3_ok = "FIGURE3" in receipts
    figure4_ok = "FIGURE4" in receipts
    table2_ok = "TABLE2" in receipts
    artifact_receipts_bound = set(receipts) == set(MAIN_ARTIFACT_KINDS)

    authorization: dict[str, Any] | None = None
    if artifact_receipts_bound and isinstance(authorization_receipt, Mapping):
        try:
            authorization = validate_run_authorization_receipt(
                authorization_receipt,
                family=family,
                authority=authority,
                artifact_receipts=receipts,
            )
        except ContractError as exc:
            evidence_errors.append(f"AUTHORIZATION_RECEIPT_INVALID:{exc}")
    explicit_run_authorization = authorization is not None
    reasons: list[str] = []
    if not complete:
        reasons.append("BLOCKED_WAITING_MAIN_MATRIX")
        state = "BLOCKED_WAITING_MAIN_MATRIX"
    elif not all((final_ok, figure3_ok, figure4_ok, table2_ok)):
        reasons.append("BLOCKED_MISSING_FINAL_MAIN_ARTIFACTS")
        state = "BLOCKED_MISSING_PROVENANCE"
    elif not explicit_run_authorization:
        reasons.append("EXPLICIT_RUN_AUTHORIZATION_REQUIRED")
        state = "READY_AFTER_MAIN_16_OF_16"
    elif not run_requested:
        reasons.append("RUN_FLAG_DISABLED")
        state = "READY_FOR_USER_APPROVAL"
    else:
        state = "AUTHORIZED_TO_BUILD_SCIENCE_PLAN"
    allowed = bool(
        complete
        and final_ok
        and figure3_ok
        and figure4_ok
        and table2_ok
        and explicit_run_authorization
        and run_requested
    )
    return LaunchGateDecision(
        state=state,
        science_launch_allowed=allowed,
        main_matrix_complete_cells=count,
        main_matrix_total_cells=EXPECTED_MAIN_CELLS,
        final_audit_pass=final_ok,
        figure3_pass=figure3_ok,
        figure4_pass=figure4_ok,
        table2_pass=table2_ok,
        explicit_run_authorization=bool(explicit_run_authorization),
        run_requested=bool(run_requested),
        authority_verified=authority_verified,
        authority_root=authority["root"],
        matrix_status_sha256=authority["matrix_status_sha256"],
        combined_audit_sha256=authority["combined_audit_sha256"],
        artifact_receipts_bound=artifact_receipts_bound,
        authorization_receipt_sha256=(
            str(authorization["authorization_sha256"])
            if authorization is not None
            else None
        ),
        evidence_errors=tuple(evidence_errors),
        reasons=tuple(reasons),
    )


__all__ = [
    "EXPECTED_MAIN_CELLS",
    "EXPECTED_MAIN_CELL_NAMES",
    "LaunchGateDecision",
    "evaluate_launch_gate",
    "load_json_object",
    "validate_matrix_authority_pointer",
]
