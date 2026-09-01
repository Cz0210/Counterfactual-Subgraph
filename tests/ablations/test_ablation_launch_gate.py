from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from src.ablations import launch_gate
from src.ablations.contracts import (
    MAIN_ARTIFACT_RECEIPT_SCHEMA,
    RUN_AUTHORIZATION_RECEIPT_SCHEMA,
    receipt_sha256,
)
from src.ablations.launch_gate import EXPECTED_MAIN_CELL_NAMES, evaluate_launch_gate
from src.ablations.status_cli import run_status
from src.eval.fast16_matrix_authority_pointer import POINTER_SCHEMA
from src.eval.four_by_four_registry import DATASETS, METHODS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    root = (tmp_path / f"authority-{count}").resolve()
    root.mkdir()
    cells = list(EXPECTED_MAIN_CELL_NAMES)
    rows = {
        (dataset, method): {
            "dataset": dataset,
            "method": method,
            "status": "FROZEN_PASS" if f"{dataset}/{method}" in cells[:count] else "INCOMPLETE",
            "standardized_output_root": str((tmp_path / "cells" / dataset / method).resolve()),
        }
        for dataset in DATASETS
        for method in METHODS
    }
    closure = {
        "root": root,
        "rows": rows,
        "complete": count,
        "matrix_sha256": "a" * 64,
        "combined_sha256": "b" * 64,
    }
    monkeypatch.setattr(launch_gate, "_verify_authority", lambda _root: closure)
    pointer = {
        "schema_version": POINTER_SCHEMA,
        "latest_authority_root": str(root),
        "latest_count": count,
        "latest_matrix_status_sha256": "a" * 64,
        "latest_combined_audit_sha256": "b" * 64,
        "applied_cells": cells[:count],
    }
    authority = {
        "root": str(root),
        "matrix_status_sha256": "a" * 64,
        "combined_audit_sha256": "b" * 64,
    }
    return pointer, authority


def _artifact_receipt(
    tmp_path: Path,
    *,
    kind: str,
    authority: dict[str, object],
) -> dict[str, object]:
    artifact = tmp_path / f"{kind.lower()}.artifact"
    artifact.write_text(f"verified {kind}\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": MAIN_ARTIFACT_RECEIPT_SCHEMA,
        "status": "PASS",
        "artifact_kind": kind,
        "matrix_authority_root": authority["root"],
        "matrix_status_sha256": authority["matrix_status_sha256"],
        "combined_audit_sha256": authority["combined_audit_sha256"],
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": _sha(artifact),
        "artifact_bytes": artifact.stat().st_size,
    }
    payload["receipt_sha256"] = receipt_sha256(payload, hash_field="receipt_sha256")
    return payload


def _authorization(
    *,
    family: str,
    authority: dict[str, object],
    receipts: dict[str, dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUN_AUTHORIZATION_RECEIPT_SCHEMA,
        "status": "AUTHORIZED",
        "authorized_by": "user_project_owner",
        "authorization_id": "future-ablation-approval-1",
        "authorized_at": "2026-09-02T00:00:00Z",
        "family": family,
        "allow_ablation_science": True,
        "run_contract_sha256": "c" * 64,
        "execution_commit": "d" * 40,
        "matrix_authority_root": authority["root"],
        "matrix_status_sha256": authority["matrix_status_sha256"],
        "combined_audit_sha256": authority["combined_audit_sha256"],
        "main_artifact_receipt_sha256s": {
            kind: receipt["receipt_sha256"]
            for kind, receipt in sorted(receipts.items())
        },
    }
    payload["authorization_sha256"] = receipt_sha256(
        payload, hash_field="authorization_sha256"
    )
    return payload


def test_ablation_gate_requires_exact_authority_and_all_bound_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, authority = _authority(tmp_path, monkeypatch, count=16)
    receipts = {
        kind: _artifact_receipt(tmp_path, kind=kind, authority=authority)
        for kind in ("FINAL_AUDIT", "FIGURE3", "FIGURE4", "TABLE2")
    }
    authorization = _authorization(family="gnn", authority=authority, receipts=receipts)
    decision = evaluate_launch_gate(
        family="gnn",
        matrix_authority=pointer,
        final_audit=receipts["FINAL_AUDIT"],
        figure3=receipts["FIGURE3"],
        figure4=receipts["FIGURE4"],
        table2=receipts["TABLE2"],
        authorization_receipt=authorization,
        run_requested=True,
    )
    assert decision.science_launch_allowed is True
    assert decision.authority_verified is True
    assert decision.artifact_receipts_bound is True
    assert decision.explicit_run_authorization is True

    fake = dict(pointer, applied_cells=[f"fake-{index}" for index in range(16)])
    rejected = evaluate_launch_gate(
        family="gnn",
        matrix_authority=fake,
        final_audit=receipts["FINAL_AUDIT"],
        figure3=receipts["FIGURE3"],
        figure4=receipts["FIGURE4"],
        table2=receipts["TABLE2"],
        authorization_receipt=authorization,
        run_requested=True,
    )
    assert rejected.science_launch_allowed is False
    assert rejected.authority_verified is False


def test_gate_rejects_boolean_style_or_drifted_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, authority = _authority(tmp_path, monkeypatch, count=16)
    receipts = {
        kind: _artifact_receipt(tmp_path, kind=kind, authority=authority)
        for kind in ("FINAL_AUDIT", "FIGURE3", "FIGURE4", "TABLE2")
    }
    decision = evaluate_launch_gate(
        family="llm",
        matrix_authority=pointer,
        final_audit=receipts["FINAL_AUDIT"],
        figure3=receipts["FIGURE3"],
        figure4=receipts["FIGURE4"],
        table2=receipts["TABLE2"],
        authorization_receipt=None,
        run_requested=True,
    )
    assert decision.science_launch_allowed is False
    assert decision.explicit_run_authorization is False

    authorization = _authorization(family="llm", authority=authority, receipts=receipts)
    authorization["run_contract_sha256"] = "e" * 64
    rejected = evaluate_launch_gate(
        family="llm",
        matrix_authority=pointer,
        final_audit=receipts["FINAL_AUDIT"],
        figure3=receipts["FIGURE3"],
        figure4=receipts["FIGURE4"],
        table2=receipts["TABLE2"],
        authorization_receipt=authorization,
        run_requested=True,
    )
    assert rejected.science_launch_allowed is False
    assert any("AUTHORIZATION_RECEIPT_INVALID" in row for row in rejected.evidence_errors)


def test_status_reads_common_config_and_defaults_run_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, _authority_evidence = _authority(tmp_path, monkeypatch, count=10)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(pointer) + "\n", encoding="utf-8")
    common = tmp_path / "common.yaml"
    common.write_text(
        "\n".join(
            (
                "schema_version: ablation_common_config_v1",
                "framework_build_only: true",
                "main_matrix_total_cells: 16",
                "explicit_run_authorization: false",
                "run_llm_ablation: false",
                "run_gnn_ablation: false",
                f"matrix_authority: {state}",
                "runtime:",
                "  gpu_lock_allowed_during_framework_build: false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    before = state.read_bytes()
    payload = run_status(
        argparse.Namespace(
            family="llm",
            common_config=common,
            matrix_authority=state,
            final_audit=None,
            figure3_pass=None,
            figure4_pass=None,
            table2_pass=None,
            authorization_receipt=None,
            run_requested=True,
            output=tmp_path / "status.json",
        )
    )
    assert state.read_bytes() == before
    assert payload["configured_run_requested"] is False
    assert payload["run_requested"] is False
    assert payload["science_started"] is False
    assert payload["gpu_lock_acquired"] is False
    assert payload["matrix_authority_mutated"] is False
