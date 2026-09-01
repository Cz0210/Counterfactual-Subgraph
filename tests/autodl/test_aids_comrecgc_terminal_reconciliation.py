from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl.reconcile_aids_comrecgc_terminal_publication import build_parser
from src.eval import aids_comrecgc_terminal_reconciliation as reconciliation
from src.eval import four_by_four_registry as registry
from src.eval import non_taste_matrix_append as matrix


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _zero_source(tmp_path: Path) -> Path:
    root = tmp_path / "science"
    standardized = root / "standardized"
    common = {
        "dataset": "AIDS",
        "dataset_key": "aids",
        "method": "COMRECGC-Adapted-DeterministicChemRepair",
        "cf_mode": "strict_flip",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "run_complete": True,
    }
    _json(standardized / "run_manifest.json", common)
    _json(standardized / "summary.json", common)
    # Production build_final_audit() intentionally has no dataset/dataset_key.
    _json(
        standardized / "final_artifact_audit.json",
        {
            "method": common["method"],
            "cf_mode": "strict_flip",
            "scientific_output_empty": True,
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
            "run_complete": True,
            "audit_passed": True,
        },
    )
    gate = root / "full_gate/gate_result.json"
    _json(
        gate,
        {
            "status": "FULL_EXECUTION_PASS",
            "audit_passed": True,
            "run_complete": True,
            "dataset": "aids",
            "scientific_output_empty": True,
            "scientific_output_status": "SCIENTIFIC_OUTPUT_EMPTY",
            "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        },
    )
    _json(
        standardized / "freeze_manifest.json",
        {
            "source_gate_result_path": str(gate.resolve()),
            "source_gate_result_sha256": _sha(gate),
        },
    )
    _json(standardized / "_FINALIZED.json", {"finalized": True})
    prefixes = [
        {
            "method": common["method"],
            "k": k,
            "close_cf_coverage": 0.0,
            "num_any_strict_flip_parents": 0,
            "conditional_mean_cost": None,
            "conditional_median_cost": None,
            "fixed_capped_mean_cost": 1.0,
            "fixed_capped_median_cost": 1.0,
        }
        for k in range(1, 21)
    ]
    _json(standardized / "prefix_metrics.json", {"prefix_metrics": prefixes})
    _csv(standardized / "prefix_metrics.csv", prefixes)
    _csv(
        standardized / "figure3_coverage_vs_k.csv",
        [
            {
                "method": common["method"],
                "k": k,
                "close_cf_coverage": 0.0,
                "conditional_mean_cost": "",
                "conditional_median_cost": "",
                "fixed_capped_mean_cost": 1.0,
                "fixed_capped_median_cost": 1.0,
            }
            for k in range(1, 21)
        ],
    )
    _csv(
        standardized / "figure4_coverage_vs_threshold.csv",
        [
            {
                "method": common["method"],
                "threshold": 0.0,
                "close_cf_coverage": 0.0,
            },
            {
                "method": common["method"],
                "threshold": 0.1,
                "close_cf_coverage": 0.0,
            },
        ],
    )
    for k in (10, 20):
        _csv(
            standardized / f"table2_comrecgc_k{k}.csv",
            [
                {
                    "k": k,
                    "method": common["method"],
                    "dataset": "AIDS",
                    "coverage": 0.0,
                    "ccrcov": 0.0,
                    "conditional_mean_cost": "",
                    "conditional_median_cost": "",
                    "fixed_capped_mean_cost": 1.0,
                    "fixed_capped_median_cost": 1.0,
                }
            ],
        )
    return root


def _science_projection(source: Path) -> dict[str, object]:
    standardized = source / "standardized"
    return {
        "root": str(source.resolve()),
        "controller_manifest_path": "/control/controller.manifest.json",
        "controller_manifest_sha256": "1" * 64,
        "exact_stage_receipt_path": "/science/exact.json",
        "exact_stage_receipt_sha256": "2" * 64,
        "final_stage_receipt_path": "/science/final.json",
        "final_stage_receipt_sha256": "3" * 64,
        "continuation_terminal_sha256": "4" * 64,
        "common_terminal_sha256": "5" * 64,
        "source_generation_root": "/science/generation",
        "source_integrity_final_sha256": "6" * 64,
        "standardized": {
            "root": str(standardized.resolve()),
            "source_evaluation_root": "/science/unified_eval",
            "run_manifest_sha256": _sha(standardized / "run_manifest.json"),
            "final_artifact_audit_sha256": _sha(
                standardized / "final_artifact_audit.json"
            ),
            "freeze_manifest_sha256": _sha(standardized / "freeze_manifest.json"),
            "identities": {"oracle_hash": "7" * 64},
        },
        "inventory": {"PASS": {"bytes": 5, "sha256": "8" * 64}},
    }


def test_zero_strict_flip_is_a_valid_non_imputed_science_result(tmp_path: Path) -> None:
    source = _zero_source(tmp_path)
    evidence = reconciliation.validate_zero_strict_flip_science(source)
    assert evidence["status"] == "PASS"
    assert evidence["scientific_output_empty"] is True
    assert evidence["coverage"] == 0.0
    assert evidence["conditional_cost_available"] is False
    assert evidence["numeric_imputation_used"] is False


@pytest.mark.parametrize("failure", ["coverage", "flip", "cost"])
def test_zero_reconciliation_rejects_nonzero_or_imputed_exports(
    tmp_path: Path, failure: str
) -> None:
    source = _zero_source(tmp_path)
    prefix = source / "standardized/prefix_metrics.json"
    value = json.loads(prefix.read_text(encoding="utf-8"))
    if failure == "coverage":
        value["prefix_metrics"][0]["close_cf_coverage"] = 0.1
    elif failure == "flip":
        value["prefix_metrics"][0]["num_any_strict_flip_parents"] = 1
    else:
        value["prefix_metrics"][0]["conditional_median_cost"] = 0.0
    _json(prefix, value)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="prefix contains",
    ):
        reconciliation.validate_zero_strict_flip_science(source)


def _controller_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    manifest_path = tmp_path / "controller.manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    controller = tmp_path / "controller"
    controller.mkdir()
    (controller / ".controller.lock").write_text("{}\n", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    manifest = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": "a" * 64,
        "controller_root": str(controller.resolve()),
    }
    state = {
        "schema_version": reconciliation.STATE_SCHEMA,
        "controller_id": reconciliation.CONTROLLER_ID,
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "status": "RUNNING",
        "current_stage": None,
        "stages": {stage: "PASS" for stage in reconciliation.STAGE_ORDER},
        "controller_process": {"pid": 321, "start_ticks": 456},
        "worker": None,
        "startup_barrier": None,
    }
    _json(controller / "state.json", state)
    for index, stage in enumerate(reconciliation.STAGE_ORDER, start=1):
        _json(controller / f"gates/{index:02d}_{stage}.json", {"gate_sha256": stage})
    return manifest_path, controller, proc, manifest


def test_missing_controller_terminal_requires_all_pass_and_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, controller, proc, manifest = _controller_fixture(tmp_path)
    monkeypatch.setattr(reconciliation, "load_bound_controller_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        reconciliation,
        "open_typed_recovery_gate",
        lambda _manifest, stage: {"gate_sha256": stage},
    )
    monkeypatch.setattr(
        reconciliation,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    result = reconciliation.validate_missing_controller_terminal(
        manifest_path, proc_root=proc
    )
    assert result["all_typed_stages_pass"] is True
    assert result["controller_terminal_present"] is False
    assert result["controller_restart_performed"] is False

    state_path = controller / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stages"][reconciliation.STAGE_ORDER[-1]] = "RUNNING"
    _json(state_path, state)
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="terminal-publication gap",
    ):
        reconciliation.validate_missing_controller_terminal(
            manifest_path, proc_root=proc
        )


def test_missing_controller_terminal_accepts_final_publication_failure_after_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, controller, proc, manifest = _controller_fixture(tmp_path)
    state_path = controller / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "BLOCKED",
            "current_stage": reconciliation.STAGE_ORDER[-1],
            "last_error": {
                "error_class": "FileExistsError",
                "message": "terminal publication interrupted",
                "recorded_at": "2026-09-01T00:00:00+00:00",
            },
        }
    )
    state["stages"][reconciliation.STAGE_ORDER[-1]] = "BLOCKED"
    _json(state_path, state)
    monkeypatch.setattr(
        reconciliation, "load_bound_controller_manifest", lambda _path: manifest
    )
    monkeypatch.setattr(
        reconciliation,
        "open_typed_recovery_gate",
        lambda _manifest, stage: {"gate_sha256": stage},
    )
    monkeypatch.setattr(
        reconciliation,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    result = reconciliation.validate_missing_controller_terminal(
        manifest_path, proc_root=proc
    )
    assert result["all_typed_stages_pass"] is True
    assert (
        result["mutable_state_projection"]
        == "FINAL_PUBLICATION_FAILURE_AFTER_TYPED_PASS"
    )


def test_missing_controller_terminal_rejects_pre_final_blocked_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, controller, proc, manifest = _controller_fixture(tmp_path)
    state_path = controller / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "BLOCKED",
            "current_stage": reconciliation.STAGE_ORDER[-2],
            "last_error": {"message": "science stage failed"},
        }
    )
    state["stages"][reconciliation.STAGE_ORDER[-2]] = "BLOCKED"
    _json(state_path, state)
    monkeypatch.setattr(
        reconciliation, "load_bound_controller_manifest", lambda _path: manifest
    )
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="terminal-publication gap",
    ):
        reconciliation.validate_missing_controller_terminal(
            manifest_path, proc_root=proc
        )


def test_missing_controller_terminal_rejects_ordinary_terminal_and_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, controller, proc, manifest = _controller_fixture(tmp_path)
    monkeypatch.setattr(reconciliation, "load_bound_controller_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        reconciliation,
        "open_typed_recovery_gate",
        lambda _manifest, stage: {"gate_sha256": stage},
    )
    monkeypatch.setattr(
        reconciliation,
        "scan_live_writers",
        lambda *_args, **_kwargs: {
            "procfs_verified": True,
            "writable_fd_count": 0,
            "writers": [],
        },
    )
    (controller / "terminal.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="ordinary AIDS controller terminal exists",
    ):
        reconciliation.validate_missing_controller_terminal(
            manifest_path, proc_root=proc
        )
    (controller / "terminal.json").unlink()
    with (controller / ".controller.lock").open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            reconciliation.AIDSComRecGCTerminalReconciliationError,
            match="lock is still held",
        ):
            reconciliation.validate_missing_controller_terminal(
                manifest_path, proc_root=proc
            )


def test_fresh_wrapper_is_hash_closed_and_detects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _zero_source(tmp_path)
    zero = reconciliation.validate_zero_strict_flip_science(source)
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    controller = {
        "status": "PASS",
        "controller_manifest_path": "/control/controller.manifest.json",
        "controller_manifest_sha256": "1" * 64,
        "controller_root": str(controller_root.resolve()),
        "controller_terminal_present": False,
        "controller_pass_marker_present": False,
        "controller_process_alive": False,
        "controller_lock_held": False,
        "all_typed_stages_pass": True,
        "typed_stage_gates": {
            stage: {"sha256": stage} for stage in reconciliation.STAGE_ORDER
        },
        "controller_restart_performed": False,
    }
    monkeypatch.setattr(
        reconciliation,
        "validate_missing_controller_terminal",
        lambda *_args, **_kwargs: dict(controller),
    )
    wrapper = tmp_path / "wrapper"
    receipt = reconciliation.publish_reconciliation(
        output_root=wrapper,
        science_projection=_science_projection(source),
        controller_evidence=controller,
        zero_evidence=zero,
        proc_root=tmp_path,
    )
    assert (wrapper / "PASS").read_bytes() == b"PASS\n"
    assert reconciliation.validate_reconciliation_root(wrapper, proc_root=tmp_path) == receipt
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="overlaps the read-only controller root",
    ):
        reconciliation.publish_reconciliation(
            output_root=controller_root / "forbidden-wrapper",
            science_projection=_science_projection(source),
            controller_evidence=controller,
            zero_evidence=zero,
            proc_root=tmp_path,
        )
    figure = source / "standardized/figure3_coverage_vs_k.csv"
    figure.write_text(figure.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        reconciliation.AIDSComRecGCTerminalReconciliationError,
        match="evidence changed",
    ):
        reconciliation.validate_reconciliation_root(wrapper, proc_root=tmp_path)


def test_matrix_dispatch_accepts_only_matching_reconciliation_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "PASS").write_bytes(b"PASS\n")
    _json(wrapper / "run_manifest.json", {"schema_version": reconciliation.RECONCILIATION_SCHEMA})
    source = tmp_path / "science"
    source.mkdir()
    controller_manifest = tmp_path / "controller.json"
    controller_manifest.write_text("{}\n", encoding="utf-8")
    science = {
        "root": str(source.resolve()),
        "standardized": {"root": str((source / "standardized").resolve())},
        "inventory": {"PASS": {}},
    }
    projection = {"root": str(source.resolve()), "static": True}
    receipt = {
        "source_science_root": str(source.resolve()),
        "controller_terminal_reconciliation": {
            "controller_manifest_path": str(controller_manifest.resolve())
        },
        "science_terminal_projection": projection,
        "reconciliation_sha256": "f" * 64,
    }
    monkeypatch.setattr(matrix, "validate_aids_reconciliation_root", lambda *_a, **_k: receipt)
    monkeypatch.setattr(
        matrix, "validate_aids_missing_controller_terminal", lambda *_a, **_k: receipt["controller_terminal_reconciliation"]
    )
    monkeypatch.setattr(matrix, "_validate_aids_science_terminal", lambda *_a, **_k: dict(science))
    monkeypatch.setattr(matrix, "aids_science_terminal_projection", lambda _value: projection)
    monkeypatch.setattr(matrix, "_critical_inventory", lambda *_a, **_k: {})
    result = matrix._validate_aids_terminal(
        wrapper,
        controller_manifest_path=controller_manifest,
        proc_root=tmp_path,
        require_writer_audit=False,
    )
    assert result["root"] == str(source.resolve())
    assert result["terminal_kind"] == "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION"
    assert result["numeric_imputation_used"] is False

    receipt["science_terminal_projection"] = {"root": str(source.resolve()), "static": False}
    with pytest.raises(matrix.NonTasteMatrixAppendError, match="projection changed"):
        matrix._validate_aids_terminal(
            wrapper,
            controller_manifest_path=controller_manifest,
            proc_root=tmp_path,
            require_writer_audit=False,
        )


def test_registry_mismatch_is_only_unavailable_conditional_cost(tmp_path: Path) -> None:
    source = _zero_source(tmp_path)
    _, reasons, k_max, table2_k, _ = registry._validate_standardized_csvs(
        source / "standardized", "ComRecGC"
    )
    assert set(reasons) == {
        "FIGURE3_INVALID:ValueError",
        "TABLE2_INVALID:ValueError",
    }
    assert k_max == 20
    assert table2_k == 10


def test_registry_promotes_only_exact_reconciled_zero_cost_mismatch() -> None:
    terminal = {
        "terminal_kind": "AIDS_ZERO_STRICT_FLIP_TERMINAL_RECONCILIATION",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "coverage": 0.0,
        "conditional_cost_available": False,
        "numeric_imputation_used": False,
        "registry_numeric_imputation_used": False,
    }
    target = {
        "dataset": "AIDS",
        "method": "ComRecGC",
        "status": registry.CellStatus.INCOMPLETE.value,
        "k_max": 20,
        "table2_k": 10,
        "adoption_reason": "",
        "rerun_reason": (
            "FIGURE3_INVALID:ValueError;TABLE2_INVALID:ValueError"
        ),
    }
    promoted = matrix._reconcile_aids_zero_registry_row(target, terminal=terminal)
    assert promoted["status"] == registry.CellStatus.FROZEN_PASS.value
    assert promoted["rerun_reason"] == ""
    assert "no numeric value was imputed" in promoted["adoption_reason"]

    target["rerun_reason"] += ";MISSING_OR_EMPTY:summary.json"
    rejected = matrix._reconcile_aids_zero_registry_row(target, terminal=terminal)
    assert rejected["status"] == registry.CellStatus.INCOMPLETE.value
    assert "MISSING_OR_EMPTY" in rejected["rerun_reason"]


def test_cli_and_slurm_keep_publication_on_shared_pointer() -> None:
    args = build_parser().parse_args(
        [
            "publish",
            "--controller-manifest",
            "/control/controller.json",
            "--reconciliation-root",
            "/runtime/reconciliation",
            "--matrix-output-root",
            "/runtime/authority-next",
        ]
    )
    assert args.authority_state_path == Path(
        "/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json"
    )
    script = Path("scripts/slurm/reconcile_aids_comrecgc_terminal_publication.sh").read_text(
        encoding="utf-8"
    )
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert token in script
