from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.autodl.run_mut_next_stage_executor_v1 as executor_cli
import scripts.autodl.run_mut_route_b_closeout_v1 as route_b_cli
from src.utils.autodl_mut_next_stage_executor_v1 import MutNextStageError
from src.utils.autodl_mut_first_divergence_v1 import stable_sha256
from src.utils.autodl_mut_successor_stages_v1 import (
    EXPORT_SCHEMA,
    LOCATOR_SCHEMA,
    MUT_CELL_ID,
    ROUTE_B_MISSING_ADAPTERS,
    MutSuccessorStageError,
    publish_canonical_mut_cell,
    reopen_completed_export,
    validate_export_receipt,
    write_route_b_adapter_blocker,
)
from src.utils.final16_owner_registry_v1 import build_owner_registry


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source-terminal"
    standardized = root / "standardized"
    standardized.mkdir(parents=True)
    for name in (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_comrecgc_k10.csv",
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
        "freeze_manifest.json",
        "_FINALIZED.json",
    ):
        (standardized / name).write_text(f"{name}\n", encoding="utf-8")
    return root


def _validator(root: Path, **_kwargs: object) -> dict[str, object]:
    return {
        "terminal_kind": "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL",
        "root": str(root.resolve()),
        "standardized": {"root": str((root / "standardized").resolve())},
        "writer_audit": {"state": "NO_ACTIVE_WRITER"},
    }


def test_strict_reopen_seals_existing_exports_without_recomputation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    output = tmp_path / "export-stage"
    terminal = reopen_completed_export(
        terminal_root=source,
        output_root=output,
        proc_root=proc,
        terminal_validator=_validator,
    )
    assert terminal["schema_version"] == EXPORT_SCHEMA
    assert terminal["status"] == "PASS"
    assert terminal["scientific_metrics_recomputed"] is False
    assert terminal["figure_table_recomputed"] is False
    assert (output / "PASS").read_bytes() == b"PASS\n"
    assert validate_export_receipt(output / "terminal.json")["receipt_sha256"]

    (source / "standardized/figure3_coverage_vs_k.csv").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(MutSuccessorStageError, match="changed after sealing"):
        validate_export_receipt(output / "terminal.json")


def test_reopen_refuses_missing_table2(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "standardized/table2_comrecgc_k10.csv").unlink()
    proc = tmp_path / "proc"
    proc.mkdir()
    with pytest.raises(MutSuccessorStageError, match="table2_comrecgc_k10"):
        reopen_completed_export(
            terminal_root=source,
            output_root=tmp_path / "export-stage",
            proc_root=proc,
            terminal_validator=_validator,
        )


def _registry(
    tmp_path: Path,
    *,
    authority: Path,
    publisher_id: str,
    locator: Path,
    lease: Path,
    commit: str,
) -> Path:
    value = build_owner_registry(
        registry_id="final16-test",
        matrix_authority_root=authority,
        tasks=[],
        publishers=[
            {
                "publisher_id": publisher_id,
                "cell_id": MUT_CELL_ID,
                "owner_state": "PREDEPLOYED",
                "owner_pid": None,
                "owner_start_ticks": None,
                "heartbeat": None,
                "locator": str(locator.resolve()),
                "lease_path": str(lease.resolve()),
                "execution_commit": commit,
                "claim_enabled": True,
                "active_writer_count": 0,
            }
        ],
        gpu_leases=[],
        check_processes=False,
    )
    return _json(tmp_path / "registry.json", value)


def test_publish_appends_once_and_then_writes_canonical_locator(tmp_path: Path) -> None:
    source = _source(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    export_root = tmp_path / "export-stage"
    reopen_completed_export(
        terminal_root=source,
        output_root=export_root,
        proc_root=proc,
        terminal_validator=_validator,
    )
    authority = tmp_path / "authority"
    authority.mkdir()
    locator = tmp_path / "control/mut.locator.json"
    lease = tmp_path / "control/mut.publisher.lock"
    commit = "a" * 40
    registry = _registry(
        tmp_path,
        authority=authority,
        publisher_id="mut-publisher",
        locator=locator,
        lease=lease,
        commit=commit,
    )
    matrix_output = tmp_path / "matrix-13"
    calls: list[str] = []

    def append_cell(**kwargs: object) -> dict[str, object]:
        calls.append("cell")
        Path(str(kwargs["output_root"])).mkdir()
        return {
            "status": "PASS",
            "output_root": str(Path(str(kwargs["output_root"])).resolve()),
            "appended_cell": MUT_CELL_ID,
            "marker": "[MATRIX_13_OF_16_PASS]",
        }

    def append_pointer(**kwargs: object) -> dict[str, object]:
        calls.append("pointer")
        assert kwargs["initial_authority_root"] is None
        return dict(kwargs["append"](tmp_path / "matrix-12"))

    terminal = publish_canonical_mut_cell(
        terminal_root=source,
        export_receipt=export_root / "terminal.json",
        owner_registry=registry,
        publisher_id="mut-publisher",
        publisher_locator=locator,
        publisher_lease_path=lease,
        matrix_authority_root=authority,
        matrix_output_root=matrix_output,
        output_root=tmp_path / "publish-stage",
        proc_root=proc,
        git_identity={"commit": commit, "tree": "b" * 40},
        terminal_validator=_validator,
        append_cell=append_cell,
        append_pointer=append_pointer,
    )
    assert calls == ["pointer", "cell"]
    assert terminal["status"] == "PASS"
    locator_payload = json.loads(locator.read_text(encoding="utf-8"))
    assert locator_payload == {
        "schema_version": LOCATOR_SCHEMA,
        "status": "READY",
        "dataset": "Mutagenicity",
        "method": "ComRecGC",
        "terminal_root": str(source.resolve()),
    }


def test_publish_rejects_another_canonical_publisher_identity(tmp_path: Path) -> None:
    source = _source(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    export_root = tmp_path / "export-stage"
    reopen_completed_export(
        terminal_root=source,
        output_root=export_root,
        proc_root=proc,
        terminal_validator=_validator,
    )
    authority = tmp_path / "authority"
    authority.mkdir()
    locator = tmp_path / "control/mut.locator.json"
    lease = tmp_path / "control/mut.publisher.lock"
    registry = _registry(
        tmp_path,
        authority=authority,
        publisher_id="other-publisher",
        locator=locator,
        lease=lease,
        commit="a" * 40,
    )
    with pytest.raises(MutSuccessorStageError, match="publisher binding changed"):
        publish_canonical_mut_cell(
            terminal_root=source,
            export_receipt=export_root / "terminal.json",
            owner_registry=registry,
            publisher_id="mut-publisher",
            publisher_locator=locator,
            publisher_lease_path=lease,
            matrix_authority_root=authority,
            matrix_output_root=tmp_path / "matrix-13",
            output_root=tmp_path / "publish-stage",
            proc_root=proc,
            git_identity={"commit": "a" * 40, "tree": "b" * 40},
            terminal_validator=_validator,
        )


def test_route_b_adapter_writes_typed_blocker_without_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision: dict[str, object] = {
        "schema_version": "mut_post_same_contract_ab_decision_v1",
        "classification": "SCIENTIFIC_STATE_DIVERGENCE",
    }
    decision["decision_sha256"] = stable_sha256(decision)
    decision_path = _json(tmp_path / "next_action.consumed.json", decision)
    monkeypatch.setattr(
        "src.utils.autodl_mut_successor_stages_v1.validate_route_b_evidence",
        lambda value, check_files: value,
    )
    output = tmp_path / "route-b"
    terminal = write_route_b_adapter_blocker(
        decision_path=decision_path,
        output_root=output,
    )
    assert terminal["status"] == "BLOCKED_ADAPTER_MISSING"
    assert terminal["fresh_50k_started"] is False
    assert terminal["route_b_started"] is False
    assert tuple(terminal["missing_adapters"]) == ROUTE_B_MISSING_ADAPTERS
    assert not (output / "PASS").exists()
    assert (output / "BLOCKED_ADAPTER_MISSING").is_file()


def test_generic_executor_preserves_false_route_b_started_on_typed_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    action = _json(tmp_path / "action/next_action.json", {"ready": True})
    predecessor = _json(tmp_path / "predecessor/terminal.json", {"status": "PASS"})
    predecessor_spec = _json(tmp_path / "predecessor/spec.json", {"sealed": True})
    raw_spec = {
        "task_id": "mut-next",
        "spec_sha256": "a" * 64,
        "runtime_root": str(tmp_path / "runtime"),
        "lease_path": str(tmp_path / "executor.lock"),
        "next_action_path": str(action),
        "predecessor_terminal": str(predecessor),
        "predecessor_task_id": "mut-ab",
        "predecessor_task_spec": str(predecessor_spec),
        "publisher_id": "mut-publisher",
        "publisher_locator": str(tmp_path / "locator.json"),
        "adoption_pipeline": [],
        "route_b_pipeline": [{"stage": "ROUTE_B", "environment": {}}],
    }
    task_spec = _json(tmp_path / "successor.json", raw_spec)
    consumed = tmp_path / "action/next_action.consumed.json"
    monkeypatch.setattr(executor_cli, "validate_successor_spec", lambda *_a, **_k: raw_spec)
    monkeypatch.setattr(
        executor_cli,
        "consume_next_action_once",
        lambda **_kwargs: (
            "ROUTE_B",
            {},
            consumed,
            {"receipt_sha256": "b" * 64},
        ),
    )
    observed_stage: dict[str, object] = {}

    def blocked_stage(stage: dict[str, object], **_kwargs: object) -> dict[str, object]:
        observed_stage.update(stage)
        return {
            "stage": "ROUTE_B",
            "terminal_status": "BLOCKED_ADAPTER_MISSING",
            "route_b_started": False,
            "fresh_50k_started": False,
        }

    monkeypatch.setattr(executor_cli, "run_stage", blocked_stage)
    with pytest.raises(MutNextStageError, match="BLOCKED_ADAPTER_MISSING"):
        executor_cli.run_executor(task_spec=task_spec, once=True)
    terminal = json.loads(
        (tmp_path / "runtime/terminal.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "BLOCKED"
    assert terminal["route_b_started"] is False
    assert terminal["fresh_50k_started"] is False
    environment = observed_stage["environment"]
    assert isinstance(environment, dict)
    assert environment["MUT_NEXT_ACTION_CONSUMED_PATH"] == str(consumed)
    assert environment["MUT_NEXT_ACTION_CONSUMPTION_RECEIPT"].endswith(
        "/runtime/next_action_consumption.json"
    )


def test_route_b_cli_uses_executor_bound_consumed_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = tmp_path / "next_action.consumed.json"
    decision.write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def blocker(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "BLOCKED_ADAPTER_MISSING"}

    monkeypatch.setenv("MUT_NEXT_ACTION_CONSUMED_PATH", str(decision.resolve()))
    monkeypatch.setattr(route_b_cli, "write_route_b_adapter_blocker", blocker)
    assert route_b_cli.main(["--output-root", str(tmp_path / "blocked")]) == 0
    assert observed["decision_path"] == decision.resolve()


def test_slurm_wrappers_follow_the_pinned_repository_contract() -> None:
    for name in (
        "reopen_mut_successor_export_v1.sh",
        "publish_mut_successor_v1.sh",
        "run_mut_route_b_closeout_v1.sh",
    ):
        text = (Path("scripts/slurm") / name).read_text(encoding="utf-8")
        for required in (
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
            assert required in text
