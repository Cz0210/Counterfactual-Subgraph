from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import tastemolnet_t11_policy_path_relocation as relocation
from src.utils.tastemolnet_research_policy import (
    NO_REDISTRIBUTION_MARKER,
    POLICY_V2_AUDIT_MARKER,
    TasteLocalDataAuthority,
    load_tastemolnet_research_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKED_POLICY = (
    PROJECT_ROOT
    / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
)


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    source_policy = tmp_path / "old-worktree" / "configs" / "taste-policy.yaml"
    current_policy = tmp_path / "new-worktree" / "configs" / "taste-policy.yaml"
    source_policy.parent.mkdir(parents=True)
    current_policy.parent.mkdir(parents=True)
    policy_bytes = TRACKED_POLICY.read_bytes()
    source_policy.write_bytes(policy_bytes)
    current_policy.write_bytes(policy_bytes)

    prepared = tmp_path / "prepared"
    cache = tmp_path / "cache"
    prepared.mkdir()
    cache.mkdir()
    authority = TasteLocalDataAuthority(
        prepared_root=prepared,
        graph_cache_root=cache,
        provenance_manifest_sha256="1" * 64,
        prepared_output_manifest_sha256="2" * 64,
        split_manifest_sha256="3" * 64,
        graph_cache_manifest_sha256="4" * 64,
        source_csv_sha256="5" * 64,
        prepared_rows=13421,
        split_rows={
            "train": 9437,
            "validation": 1328,
            "calibration": 1328,
            "test": 1328,
        },
        graph_cache_rows=13421,
    )
    monkeypatch.setattr(
        relocation,
        "validate_tastemolnet_local_authority",
        lambda policy, *, prepared_root, graph_cache_root: authority,
    )

    source = load_tastemolnet_research_policy(source_policy)
    source.require_main_route()
    audit = tmp_path / "policy-audit"
    audit.mkdir()
    receipt = audit / "tastemolnet_policy_receipt.json"
    _json(
        receipt,
        {
            "schema_version": "tastemolnet_research_reporting_policy_receipt_v2",
            "created_at": "2026-09-02T00:00:00+00:00",
            "dataset": "tastemolnet",
            "status": source.status,
            "authorization_state": source.authorization_state,
            "authorization_status": source.authorization_status,
            "policy": source.evidence(),
            "private_data_authority": authority.evidence(),
            "run_tastemolnet": 1,
            "heavy_route_authorized": True,
            "paper_reporting_authorized": True,
            "dataset_redistribution_authorized": False,
            "upstream_terms_status": "NOT_EXPLICITLY_STATED",
            "license_conclusion": "NOT_GRANTED_OR_INFERRED",
            "hpc_execution_authorized": False,
            "data_reprepared": False,
            "graph_cache_rebuilt": False,
            "terminal_marker": POLICY_V2_AUDIT_MARKER,
            "no_redistribution_marker": NO_REDISTRIBUTION_MARKER,
        },
    )
    (audit / "tastemolnet_policy_audit.md").write_text("audit\n", encoding="utf-8")
    (audit / POLICY_V2_AUDIT_MARKER).write_text(
        POLICY_V2_AUDIT_MARKER + "\n", encoding="utf-8"
    )
    (audit / NO_REDISTRIBUTION_MARKER).write_text(
        NO_REDISTRIBUTION_MARKER + "\n", encoding="utf-8"
    )
    return source_policy, current_policy, receipt, prepared, cache


def test_build_and_reopen_relocation_after_old_checkout_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, current, receipt, prepared, cache = _fixture(tmp_path, monkeypatch)
    overlay = tmp_path / "publication-reconciliation" / "t11-policy-relocation"
    built = relocation.build_t11_policy_path_relocation(
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_root=overlay,
    )
    assert built.path == overlay / relocation.RELOCATION_FILENAME
    assert built.payload["source_policy_path"] == str(source)
    assert built.payload["current_policy_path"] == str(current)
    assert built.payload["source_policy_sha256"] == built.payload[
        "current_policy_sha256"
    ]
    assert built.policy.policy_id == built.payload["policy_id"]
    assert built.publication_evidence()["only_policy_path_relocated"] is True

    # Retry logs are operational evidence, not part of the relocation receipt's
    # scientific identity and therefore do not change receipt validation.
    (overlay / "append.stdout").write_bytes(b"")
    (overlay / "append.stderr").write_text("prior failed retry\n", encoding="utf-8")
    source.unlink()
    reopened = relocation.validate_t11_policy_path_relocation(
        built.path,
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
    )
    assert reopened.sha256 == built.sha256
    assert reopened.policy.canonical_sha256 == built.policy.canonical_sha256


def test_relocation_fails_if_current_raw_policy_bytes_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, current, receipt, prepared, cache = _fixture(tmp_path, monkeypatch)
    overlay = tmp_path / "overlay"
    built = relocation.build_t11_policy_path_relocation(
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_root=overlay,
    )
    current.write_bytes(current.read_bytes() + b"\n# byte drift\n")
    with pytest.raises(
        relocation.T11PolicyPathRelocationError, match="policy_receipt.policy"
    ):
        relocation.validate_t11_policy_path_relocation(
            built.path,
            current_policy_path=current,
            policy_receipt=receipt,
            prepared_root=prepared,
            graph_cache_root=cache,
        )


def test_relocation_fails_on_non_path_source_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, current, receipt, prepared, cache = _fixture(tmp_path, monkeypatch)
    overlay = tmp_path / "overlay"
    built = relocation.build_t11_policy_path_relocation(
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_root=overlay,
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["paper_reporting_authorized"] = False
    _json(receipt, payload)
    with pytest.raises(
        relocation.T11PolicyPathRelocationError, match="typed Taste policy receipt"
    ):
        relocation.validate_t11_policy_path_relocation(
            built.path,
            current_policy_path=current,
            policy_receipt=receipt,
            prepared_root=prepared,
            graph_cache_root=cache,
        )


def test_relocation_fails_on_overlay_tamper_or_native_type_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, current, receipt, prepared, cache = _fixture(tmp_path, monkeypatch)
    built = relocation.build_t11_policy_path_relocation(
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_root=tmp_path / "overlay",
    )
    payload = json.loads(built.path.read_text(encoding="utf-8"))
    payload["policy_version"] = 2.0
    _json(built.path, payload)
    with pytest.raises(
        relocation.T11PolicyPathRelocationError, match="native JSON type"
    ):
        relocation.validate_t11_policy_path_relocation(
            built.path,
            current_policy_path=current,
            policy_receipt=receipt,
            prepared_root=prepared,
            graph_cache_root=cache,
        )


def test_relocation_output_is_fresh_and_does_not_touch_policy_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, current, receipt, prepared, cache = _fixture(tmp_path, monkeypatch)
    before = {
        path: path.read_bytes() for path in (source, current, receipt)
    }
    overlay = tmp_path / "overlay"
    relocation.build_t11_policy_path_relocation(
        current_policy_path=current,
        policy_receipt=receipt,
        prepared_root=prepared,
        graph_cache_root=cache,
        output_root=overlay,
    )
    with pytest.raises(
        relocation.T11PolicyPathRelocationError, match="must be fresh"
    ):
        relocation.build_t11_policy_path_relocation(
            current_policy_path=current,
            policy_receipt=receipt,
            prepared_root=prepared,
            graph_cache_root=cache,
            output_root=overlay,
        )
    assert {path: path.read_bytes() for path in before} == before
