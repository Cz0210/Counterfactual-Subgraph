from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.build_tastemolnet_multiclass_baseline_tasks import (
    _absolute_file,
    main,
)
from src.baselines.tastemolnet_multiclass_adapters import (
    TasteMulticlassContractError,
)
from src.baselines.tastemolnet_multiclass_tasks import (
    BLOCKER,
    FRAGMENT_SCHEMA,
    METHODS,
    build_blocked_fragment,
    release_contract,
)


def _blocked_gate() -> dict[str, object]:
    return {
        "schema_version": "tastemolnet_license_audit_v1",
        "dataset": "tastemolnet",
        "status": BLOCKER,
        "heavy_route_authorized": False,
        "run_tastemolnet": False,
        "blocked_reason": BLOCKER,
    }


def _pass_gate() -> dict[str, object]:
    return {
        "schema_version": "tastemolnet_license_audit_v1",
        "dataset": "tastemolnet",
        "status": "PASS",
        "heavy_route_authorized": True,
        "run_tastemolnet": True,
        "approval_evidence": {"sha256": "a" * 64},
    }


def test_blocked_fragment_has_three_non_runnable_zero_gpu_routes() -> None:
    payload = build_blocked_fragment(license_gate=_blocked_gate())
    assert payload["schema_version"] == FRAGMENT_SCHEMA
    assert payload["status"] == BLOCKER
    assert payload["heavy_route_authorized"] is False
    tasks = payload["tasks"]
    assert len(tasks) == len(METHODS)
    assert {task["id"] for task in tasks} == {
        "tastemolnet_gcfexplainer",
        "tastemolnet_globalgce",
        "tastemolnet_comrecgc",
    }
    for task in tasks:
        assert task["depends_on"] == ["tastemolnet_license_audit"]
        assert task["resource"] == "cpu"
        assert task["manifest_only"] is True
        assert task["data_splits"] == []
        assert task["command"] is None
        assert task["blocked_reason"] == BLOCKER
        assert task["heavy_route_authorized"] is False
        assert task["run_tastemolnet"] is False


def test_release_contract_requires_all_shared_multiclass_gates() -> None:
    contract = release_contract()
    serialized = json.dumps(contract, sort_keys=True)
    assert contract["release_mode"] == "new_fresh_fragment_only"
    assert contract["blocked_fragment_mutation_forbidden"] is True
    assert '"status": "PASS"' in serialized
    assert '"num_classes": 3' in serialized
    assert '"source_label": 1' in serialized
    assert '"rf_oracle_used": false' in serialized
    assert '"target_branches": [0, 2]' in serialized
    assert '"test": "after_frozen_selector_only"' in serialized
    assert (
        contract["method_extensions"]["ComRecGC"]["graph_content_identity"]
        == "canonical_global_graph_hash"
    )


def test_blocked_builder_refuses_to_relabel_a_pass_gate() -> None:
    with pytest.raises(TasteMulticlassContractError, match="new runnable fragment"):
        build_blocked_fragment(license_gate=_pass_gate())


def test_cli_writes_fresh_blocked_fragment_and_never_heavy(tmp_path: Path) -> None:
    gate = tmp_path / "taste_license_gate.json"
    gate.write_text(json.dumps(_blocked_gate()), encoding="utf-8")
    output = tmp_path / "fragments/taste-baselines.json"
    assert main(["--output", str(output), "--license-gate", str(gate)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == BLOCKER
    assert payload["license_evidence"]["gate_path"] == str(gate.resolve())
    assert payload["license_evidence"]["gate_hash"]
    assert all(task["command"] is None for task in payload["tasks"])
    with pytest.raises(FileExistsError, match="must be fresh"):
        main(["--output", str(output), "--license-gate", str(gate)])


def test_cli_refuses_symlinked_license_gate(tmp_path: Path) -> None:
    physical = tmp_path / "physical.json"
    physical.write_text(json.dumps(_blocked_gate()), encoding="utf-8")
    symlink = tmp_path / "gate.json"
    symlink.symlink_to(physical)
    with pytest.raises(Exception, match="physical license gate"):
        _absolute_file(str(symlink))
