from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl import append_bace_gcf_matrix_authority as append_module
from src.eval.four_by_four_registry import (
    AuditConfig,
    RegistryResult,
    audit_registry,
    write_registry_outputs,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _template(tmp_path: Path) -> RegistryResult:
    scan = tmp_path / "empty-scan"
    scan.mkdir(parents=True)
    return audit_registry(AuditConfig(scan_roots=(scan,), output_root=tmp_path / "unused"))


def _with_rows(template: RegistryResult, rows: list[dict[str, object]], count: int) -> RegistryResult:
    return RegistryResult(
        matrix_rows=tuple(rows),
        inventory_rows=template.inventory_rows,
        stale_rows=template.stale_rows,
        oracle_registry=template.oracle_registry,
        evaluation_contract=template.evaluation_contract,
        threshold_contracts=template.threshold_contracts,
        matrix_complete_cells=count,
    )


def _authority_fixture(tmp_path: Path) -> tuple[RegistryResult, RegistryResult, Path, Path]:
    template = _template(tmp_path)
    prior_rows = [dict(row) for row in template.matrix_rows]
    pass_keys = {
        ("AIDS", "Ours"),
        ("AIDS", "GCFExplainer"),
        ("AIDS", "GlobalGCE"),
        ("Mutagenicity", "Ours"),
        ("Mutagenicity", "GCFExplainer"),
        ("Mutagenicity", "GlobalGCE"),
        ("BACE", "Ours"),
    }
    bace_identity = {
        "dataset_hash": _sha("bace-dataset"),
        "split_hash": _sha("bace-test"),
        "oracle_backend": "gnn",
        "oracle_checkpoint": "/frozen/bace/gine",
        "oracle_hash": _sha("bace-gine"),
        "molclr_checkpoint_hash": _sha("molclr"),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "threshold_config_hash": _sha("threshold-grid"),
    }
    for row in prior_rows:
        key = (str(row["dataset"]), str(row["method"]))
        if key not in pass_keys:
            continue
        source = tmp_path / "prior-cells" / key[0] / key[1]
        source.mkdir(parents=True)
        (source / "PASS").write_text("PASS\n", encoding="utf-8")
        row["status"] = "FROZEN_PASS" if key == ("BACE", "Ours") else "ADOPTABLE_PASS"
        row["standardized_output_root"] = str(source.resolve())
        row["rerun_reason"] = ""
        row["adoption_reason"] = "frozen fixture"
        if key == ("BACE", "Ours"):
            row.update(bace_identity)
    prior = _with_rows(template, prior_rows, 7)

    target_root = tmp_path / "bace-gcf-standardized"
    target_root.mkdir()
    (target_root / "PASS").write_text("PASS\n", encoding="utf-8")
    new_rows = [dict(row) for row in prior_rows]
    target = next(
        row
        for row in new_rows
        if row["dataset"] == "BACE" and row["method"] == "GCFExplainer"
    )
    target.update(
        {
            **bace_identity,
            "status": "FROZEN_PASS",
            "standardized_output_root": str(target_root.resolve()),
            "raw_output_root": str(target_root.resolve()),
            "k_max": 20,
            "table2_k": 10,
            "rerun_reason": "",
            "adoption_reason": "all required frozen evidence and protocol checks passed",
        }
    )
    current = _with_rows(template, new_rows, 8)
    return prior, current, target_root, tmp_path / "prior-authority"


def test_registry_writer_hash_closes_supplemental_outputs(tmp_path: Path) -> None:
    result = _template(tmp_path)
    output = write_registry_outputs(
        result,
        tmp_path / "registry",
        supplemental_outputs={"append_authority.json": b"{}\n"},
    )
    combined = json.loads((output / "combined_audit.json").read_text(encoding="utf-8"))
    identity = combined["files"]["append_authority.json"]
    assert identity == {
        "bytes": 3,
        "sha256": hashlib.sha256(b"{}\n").hexdigest(),
    }
    assert (output / "matrix_status.json").is_file()

    with pytest.raises(ValueError, match="reserved"):
        write_registry_outputs(
            result,
            tmp_path / "reserved",
            supplemental_outputs={"matrix_status.json": b"{}\n"},
        )
    with pytest.raises(ValueError, match="Invalid supplemental"):
        write_registry_outputs(
            result,
            tmp_path / "unsafe",
            supplemental_outputs={"../escape.json": b"{}\n"},
        )


def test_strict_append_publishes_eight_and_records_supersession(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior, current, target_root, prior_root = _authority_fixture(tmp_path)
    write_registry_outputs(prior, prior_root)
    stale_template = _template(tmp_path / "stale-fixture")
    stale_root = tmp_path / "stale-authority"
    write_registry_outputs(stale_template, stale_root)
    legacy_root = tmp_path / "legacy-top-level-snapshot"
    legacy_root.mkdir()
    (legacy_root / "matrix_status.json").write_bytes(
        (stale_root / "matrix_status.json").read_bytes()
    )
    stale_before = {
        path.relative_to(stale_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stale_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(append_module, "audit_registry", lambda _config: current)
    monkeypatch.setattr(
        append_module,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "scanned_process_count": 4,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    output = tmp_path / "authority-8of16"
    result = append_module.append_bace_gcf_authority(
        prior_authority_root=prior_root,
        bace_gcf_standardized_root=target_root,
        output_root=output,
        superseded_audit_roots=(stale_root, legacy_root),
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
    )

    assert result["matrix_complete_cells"] == 8
    matrix = json.loads((output / "matrix_status.json").read_text(encoding="utf-8"))
    assert matrix["matrix_complete_cells"] == 8
    append_receipt = json.loads((output / "append_authority.json").read_text(encoding="utf-8"))
    assert append_receipt["unchanged_prior_passing_cells"] == 7
    assert append_receipt["appended_cell"]["status"] == "FROZEN_PASS"
    supersession = json.loads(
        (output / "superseded_snapshots.json").read_text(encoding="utf-8")
    )
    assert supersession["superseded_snapshot_count"] == 2
    assert supersession["superseded_snapshots"][0]["observed_matrix_complete_cells"] == 0
    assert supersession["superseded_snapshots"][0]["historical_root_modified"] is False
    assert (
        supersession["superseded_snapshots"][1]["closure_status"]
        == "LEGACY_MATRIX_STATUS_ONLY_COMBINED_AUDIT_ABSENT"
    )
    combined = json.loads((output / "combined_audit.json").read_text(encoding="utf-8"))
    assert "append_authority.json" in combined["files"]
    assert "superseded_snapshots.json" in combined["files"]
    assert stale_before == {
        path.relative_to(stale_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stale_root.rglob("*")
        if path.is_file()
    }


def test_strict_append_rejects_any_non_target_row_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior, current, target_root, prior_root = _authority_fixture(tmp_path)
    write_registry_outputs(prior, prior_root)
    drifted_rows = [dict(row) for row in current.matrix_rows]
    aids_ours = next(
        row for row in drifted_rows if row["dataset"] == "AIDS" and row["method"] == "Ours"
    )
    aids_ours["adoption_reason"] = "changed"
    drifted = _with_rows(current, drifted_rows, 8)
    monkeypatch.setattr(append_module, "audit_registry", lambda _config: drifted)
    monkeypatch.setattr(
        append_module,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "scanned_process_count": 1,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(append_module.MatrixAppendError, match="Non-target matrix row drifted"):
        append_module.append_bace_gcf_authority(
            prior_authority_root=prior_root,
            bace_gcf_standardized_root=target_root,
            output_root=output,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
        )
    assert not output.exists()
