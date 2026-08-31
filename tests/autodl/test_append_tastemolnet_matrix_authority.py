from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.autodl.append_tastemolnet_matrix_authority import _cells
from src.eval.four_by_four_registry import (
    AuditConfig,
    audit_registry,
    stable_json_sha256,
    write_registry_outputs,
)
from src.eval.tastemolnet_matrix_append import (
    TasteMatrixAppendError,
    append_tastemolnet_cells,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _paper_csvs(root: Path, *, method: str) -> None:
    _csv(
        root / "figure3_coverage_vs_k.csv",
        ["method", "k", "coverage", "cost"],
        [
            {"method": method, "k": k, "coverage": k / 100, "cost": 0.1}
            for k in range(1, 21)
        ],
    )
    _csv(
        root / "figure4_coverage_vs_threshold.csv",
        ["method", "threshold", "coverage"],
        [
            {"method": method, "threshold": value, "coverage": value}
            for value in (0.1, 0.2, 0.3)
        ],
    )
    slug = "".join(character.lower() for character in method if character.isalnum())
    _csv(
        root / f"table2_{slug}_k10.csv",
        ["method", "k", "coverage", "cost", "flip_rate", "cf_drop"],
        [
            {
                "method": method,
                "k": 10,
                "coverage": 0.1,
                "cost": 0.1,
                "flip_rate": 0.2,
                "cf_drop": 0.3,
            }
        ],
    )


def _legacy_cell(root: Path, *, dataset: str, method: str) -> None:
    root.mkdir(parents=True)
    _paper_csvs(root, method=method)
    (root / "pair_details.csv").write_text("parent_id,candidate_id\np,c\n", encoding="utf-8")
    common = {
        "dataset": dataset,
        "method": method,
        "dataset_hash": _sha(f"{dataset}-data"),
        "test_split_hash": _sha(f"{dataset}-test"),
        "test_parent_ids_sha256": _sha(f"{dataset}-parents"),
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "oracle_checkpoint": str(root / "oracle.pkl"),
        "oracle_hash": _sha(f"{dataset}-oracle"),
        "molclr_checkpoint_hash": _sha(f"{dataset}-molclr"),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "rf_oracle_used": True,
        "raw_output_complete": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }
    (root / "oracle.pkl").write_bytes(b"oracle")
    _json(root / "summary.json", common)
    _json(root / "run_manifest.json", {**common, "raw_output_root": str(root)})
    _json(
        root / "final_artifact_audit.json",
        {**common, "passed": True, "audit_passed": True, "frozen": True},
    )


def _prior_eight(root: Path) -> Path:
    explicit: dict[str, str] = {}
    for dataset in ("AIDS", "Mutagenicity"):
        for method in ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"):
            cell = root / "cells" / dataset / method
            _legacy_cell(cell, dataset=dataset, method=method)
            explicit[f"{dataset}/{method}"] = str(cell)
    result = audit_registry(
        AuditConfig(scan_roots=(), output_root=root / "authority", explicit_cells=explicit)
    )
    assert result.matrix_complete_cells == 8
    return write_registry_outputs(result, root / "authority")


def _t3(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "t3" / "artifacts" / "checkpoint"
    checkpoint.mkdir(parents=True)
    temperature = checkpoint / "temperature_scaling.json"
    _json(temperature, {"temperature": 1.5})
    return {
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": _sha("t3-model"),
        "temperature": 1.5,
        "temperature_scaling_sha256": hashlib.sha256(temperature.read_bytes()).hexdigest(),
    }


def _policy(tmp_path: Path) -> dict[str, object]:
    receipt = tmp_path / "policy" / "tastemolnet_policy_receipt.json"
    _json(receipt, {"fixture": True})
    return {
        "policy_receipt_path": str(receipt),
        "policy_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "policy_id": "fixture",
        "policy_version": 2,
        "paper_reporting_authorized": True,
        "dataset_redistribution_authorized": False,
        "license_conclusion": "NOT_GRANTED_OR_INFERRED",
        "legacy_license_pass_claimed": False,
    }


def _taste_cell(root: Path, *, method: str, t3: dict[str, object]) -> None:
    root.mkdir(parents=True)
    _paper_csvs(root, method=method)
    for name in (
        "prefix_metrics.csv",
        "parent_best_distances.csv",
        "destination_distribution.csv",
    ):
        (root / name).write_text("method,value\n%s,1\n" % method, encoding="utf-8")
    _json(root / "prefix_metrics.json", [{"method": method, "k": 1}])
    (root / "pair_details.csv").write_text("parent_id,candidate_id\np,c\n", encoding="utf-8")
    stages = {
        "Ours": (
            "T11_OURS_FULL",
            "tastemolnet_t11_final_run_manifest_v2",
            "tastemolnet_t11_final_artifact_audit_v2",
            "[TASTE_OURS_PASS]\n",
        ),
        "GCFExplainer": (
            "T12_GCF_FULL",
            "tastemolnet_t12_final_run_manifest_v1",
            "tastemolnet_t12_terminal_verification_v1",
            "[TASTE_GCF_PASS]\n",
        ),
        "GlobalGCE": (
            "T13_GLOBALGCE_FULL",
            "tastemolnet_t13_run_manifest_v1",
            "tastemolnet_t13_terminal_verification_v1",
            "PASS\n",
        ),
        "ComRecGC": (
            "T14_COMRECGC_FULL_POSTPROCESS",
            "tastemolnet_t14_postprocess_run_manifest_v1",
            "tastemolnet_t14_postprocess_terminal_verification_v1",
            "[TASTE_COMRECGC_PASS]\n",
        ),
    }
    stage, run_schema, audit_schema, marker = stages[method]
    common = {
        "dataset": "TasteMolNet",
        "method": method,
        "stage": stage,
        "num_classes": 3,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": t3["checkpoint_dir"],
        "oracle_hash": t3["checkpoint_id"],
        "oracle_checkpoint_hash": t3["checkpoint_id"],
        "temperature_calibration_hash": t3["temperature_scaling_sha256"],
        "dataset_hash": _sha("taste-data"),
        "test_split_hash": _sha("taste-test"),
        "test_parent_ids_sha256": _sha("taste-parents"),
        "molclr_checkpoint_hash": _sha("taste-molclr"),
        "threshold_config_hash": stable_json_sha256([0.1, 0.2, 0.3]),
        "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip",
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_complete": True,
        "raw_output_root": str(root),
        "frozen": True,
    }
    _json(
        root / "summary.json",
        {**common, "status": "PASS", "destination_labels": [0, 2]},
    )
    _json(
        root / "oracle_manifest.json",
        {
            **common,
            "same_frozen_gine_for_generation_calibration_test": True,
            "temperature": 1.5,
        },
    )
    evaluation = {
        **common,
        "destination_labels": [0, 2],
        "full_cartesian_test_pairs": True,
    }
    if method == "Ours":
        evaluation.pop("destination_labels")
    _json(
        root / "evaluation_manifest.json",
        evaluation,
    )
    frozen_names = [
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
    ]
    inventory = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
        for name in frozen_names
    }
    _json(
        root / "freeze_manifest.json",
        {"files": inventory, "inventory_sha256": stable_json_sha256(inventory)},
    )
    science_proof: dict[str, object] = {}
    if method == "Ours":
        science = root.parent / "t11-science"
        science.mkdir()
        for name in frozen_names:
            shutil.copyfile(root / name, science / name)
        shutil.copyfile(root / "freeze_manifest.json", science / "freeze_manifest.json")
        final_summary = json.loads((root / "summary.json").read_text())
        final_summary["independent_terminal_verification_passed"] = True
        _json(root / "summary.json", final_summary)
        science_proof = {
            "science_root": str(science),
            "science_freeze_manifest_sha256": hashlib.sha256(
                (science / "freeze_manifest.json").read_bytes()
            ).hexdigest(),
        }
    terminal_proof = (
        {"terminal_verifier": "separate_verify_only_invocation"}
        if method == "GlobalGCE"
        else {"independent_terminal_verification_passed": True}
    )
    _json(
        root / "run_manifest.json",
        {
            "schema_version": run_schema,
            **common,
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "worker_wrote_pass": False,
            **terminal_proof,
            **science_proof,
        },
    )
    audit_common = dict(common)
    if method in {"GCFExplainer", "GlobalGCE", "ComRecGC"}:
        audit_common.pop("selection_frozen_before_test")
    if method == "Ours":
        audit_common.pop("threshold_fitted_on_test")
    audit_payload = {
            "schema_version": audit_schema,
            **audit_common,
            "status": "PASS",
            "passed": True,
            "audit_passed": True,
            "independent_verifier": method != "GlobalGCE",
            "recomputed_metrics": method == "Ours",
            "calibration_pair_chunks_replayed": method == "Ours",
            "checks": {
                "selection_frozen_before_test": True,
                "calibration_only_selector": method == "GlobalGCE",
                "calibration_only_selector_replayed": method
                in {"GCFExplainer", "ComRecGC"},
            },
        }
    if method == "Ours":
        final_names = [*frozen_names, "freeze_manifest.json", "run_manifest.json"]
        audit_payload["files"] = {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name in final_names
        }
    _json(root / "final_artifact_audit.json", audit_payload)
    (root / "PASS").write_text(marker, encoding="utf-8")


def _append(
    tmp_path: Path,
    *,
    method: str,
    prior: Path,
    destination: Path,
    t3: dict[str, object],
) -> dict[str, object]:
    cell = tmp_path / f"taste-{method}"
    _taste_cell(cell, method=method, t3=t3)
    return append_tastemolnet_cells(
        prior_authority_root=prior,
        taste_cells={method: cell},
        output_root=destination,
        t3_root=tmp_path / "unused-t3",
        policy_path=tmp_path / "unused-policy",
        policy_receipt=tmp_path / "unused-receipt",
        prepared_root=tmp_path / "unused-prepared",
        graph_cache_root=tmp_path / "unused-cache",
        proc_root=tmp_path / "unused-proc",
        require_writer_audit=False,
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
        t3_binding=t3,
        policy_binding=_policy(tmp_path),
    )


def test_strict_taste_append_preserves_non_target_rows_and_advances_once(tmp_path: Path) -> None:
    prior = _prior_eight(tmp_path)
    t3 = _t3(tmp_path)
    before = json.loads((prior / "matrix_status.json").read_text())
    result = _append(
        tmp_path,
        method="Ours",
        prior=prior,
        destination=tmp_path / "matrix-nine",
        t3=t3,
    )
    after = json.loads(Path(result["matrix_status_path"]).read_text())
    assert result["matrix_complete_cells"] == 9
    assert result["marker"] == "[MATRIX_9_OF_16_PASS]"
    before_rows = {(row["dataset"], row["method"]): row for row in before["cells"]}
    after_rows = {(row["dataset"], row["method"]): row for row in after["cells"]}
    for key, row in before_rows.items():
        if key != ("TasteMolNet", "Ours"):
            assert after_rows[key] == row
    assert after_rows[("TasteMolNet", "Ours")]["status"] == "FROZEN_PASS"


def test_append_rejects_a_smoke_or_method_marker_masquerading_as_full(tmp_path: Path) -> None:
    prior = _prior_eight(tmp_path)
    t3 = _t3(tmp_path)
    cell = tmp_path / "taste-Ours"
    _taste_cell(cell, method="Ours", t3=t3)
    (cell / "PASS").write_text("[TASTE_T6_OURS_PPO_SMOKE_PASS]\n", encoding="utf-8")
    with pytest.raises(TasteMatrixAppendError, match="method-cell PASS marker"):
        append_tastemolnet_cells(
            prior_authority_root=prior,
            taste_cells={"Ours": cell},
            output_root=tmp_path / "rejected",
            t3_root=tmp_path / "unused",
            policy_path=tmp_path / "unused",
            policy_receipt=tmp_path / "unused",
            prepared_root=tmp_path / "unused",
            graph_cache_root=tmp_path / "unused",
            proc_root=tmp_path / "unused",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
            t3_binding=t3,
            policy_binding=_policy(tmp_path),
        )


def test_batch_append_accepts_each_exact_full_terminal_contract(tmp_path: Path) -> None:
    prior = _prior_eight(tmp_path)
    t3 = _t3(tmp_path)
    cells: dict[str, Path] = {}
    for method in ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"):
        cell = tmp_path / f"taste-{method}"
        _taste_cell(cell, method=method, t3=t3)
        cells[method] = cell
    result = append_tastemolnet_cells(
        prior_authority_root=prior,
        taste_cells=cells,
        output_root=tmp_path / "matrix-twelve",
        t3_root=tmp_path / "unused",
        policy_path=tmp_path / "unused",
        policy_receipt=tmp_path / "unused",
        prepared_root=tmp_path / "unused",
        graph_cache_root=tmp_path / "unused",
        proc_root=tmp_path / "unused",
        require_writer_audit=False,
        git_identity={"commit": "a" * 40, "tree": "b" * 40},
        t3_binding=t3,
        policy_binding=_policy(tmp_path),
    )
    assert result["matrix_complete_cells"] == 12
    assert result["marker"] == "[MATRIX_12_OF_16_PASS]"


def test_append_rejects_temperature_hash_not_bound_to_t3(tmp_path: Path) -> None:
    prior = _prior_eight(tmp_path)
    t3 = _t3(tmp_path)
    cell = tmp_path / "taste-GlobalGCE"
    _taste_cell(cell, method="GlobalGCE", t3=t3)
    manifest = json.loads((cell / "run_manifest.json").read_text())
    manifest["temperature_calibration_hash"] = "f" * 64
    _json(cell / "run_manifest.json", manifest)
    with pytest.raises(TasteMatrixAppendError, match="temperature calibration hash"):
        append_tastemolnet_cells(
            prior_authority_root=prior,
            taste_cells={"GlobalGCE": cell},
            output_root=tmp_path / "rejected",
            t3_root=tmp_path / "unused",
            policy_path=tmp_path / "unused",
            policy_receipt=tmp_path / "unused",
            prepared_root=tmp_path / "unused",
            graph_cache_root=tmp_path / "unused",
            proc_root=tmp_path / "unused",
            require_writer_audit=False,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
            t3_binding=t3,
            policy_binding=_policy(tmp_path),
        )


def test_cli_cell_parser_rejects_duplicates_and_accepts_batch(tmp_path: Path) -> None:
    parsed = _cells([f"Ours={tmp_path / 'ours'}", f"GlobalGCE={tmp_path / 'global'}"])
    assert list(parsed) == ["Ours", "GlobalGCE"]
    with pytest.raises(argparse.ArgumentTypeError):  # type: ignore[name-defined]
        _cells([f"Ours={tmp_path / 'a'}", f"Ours={tmp_path / 'b'}"])
