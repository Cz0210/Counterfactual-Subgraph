from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.tastemolnet_gcf_full_postprocess import (
    METHOD,
    PASS_MARKER,
    RUN_MANIFEST_SCHEMA,
    SELECTION_SCHEMA,
    STAGE,
    VERIFY_SCHEMA,
    _authorize_test,
    _pair_rows_for_parent,
    _publish_terminal_pass,
    _validate_generation_pass,
    _validate_fullgraph_pairs,
    evaluate_split_resumable,
    select_on_calibration,
    standardized_metrics,
    TasteGCFPostprocessError,
)
from src.baselines.tastemolnet_globalgce_full import atomic_json, sha256_file
from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS
from src.eval.tastemolnet_ours_full import ThresholdContract


class _Distance:
    def distance(self, _left: str, right: str) -> dict[str, object]:
        return {
            "ok": True,
            "distance": float(int(right.rsplit("-", 1)[1]) + 1) / 100.0,
            "error": None,
        }


def _candidate(index: int) -> dict[str, object]:
    candidate_id = f"{index:064x}"
    return {
        "candidate_id": candidate_id,
        "candidate_content_hash": f"{index + 100:064x}",
        "rule_content_hash": f"{index + 100:064x}",
        "canonical_smiles": f"candidate-{index}",
        "predicted_label": 0 if index % 2 == 0 else 2,
        "probabilities": [0.8, 0.1, 0.1] if index % 2 == 0 else [0.1, 0.1, 0.8],
        "oracle_checkpoint_hash": "a" * 64,
    }


def _authority() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_id="a" * 64,
        temperature_calibration_hash="b" * 64,
        feature_schema_hash="c" * 64,
        molclr_checkpoint_sha256="d" * 64,
        threshold=SimpleNamespace(config_hash="e" * 64),
    )


def test_fullgraph_pair_semantics_selector_and_standardized_exports() -> None:
    candidates = [_candidate(index) for index in range(10)]
    parents = [
        SimpleNamespace(parent_id="p0", smiles="parent-0", split="calibration", label=1),
        SimpleNamespace(parent_id="p1", smiles="parent-1", split="calibration", label=1),
    ]
    calibration = []
    for parent in parents:
        calibration.extend(
            _pair_rows_for_parent(
                parent=parent,
                before={"predicted_label": 1, "probabilities": [0.1, 0.8, 0.1]},
                candidates=candidates,
                provider=_Distance(),
                authority=_authority(),
                split="calibration",
            )
        )
    _validate_fullgraph_pairs(calibration, candidates, split="calibration")
    assert all(row["action_kind"] == "full_counterfactual_graph" for row in calibration)
    assert all("residual_smiles" not in row for row in calibration)
    selected, selection = select_on_calibration(
        candidates, calibration, theta_star=0.2
    )
    assert len(selected) == 10
    assert selection["calibration_only"] is True
    assert selection["test_used_for_selection"] is False

    test_rows = [{**row, "split": "test"} for row in calibration]
    threshold = ThresholdContract(
        values=(0.1, 0.2, 0.3),
        theta_star=0.2,
        cost_cap=0.3,
        config_hash="e" * 64,
        source="frozen",
        source_split="calibration",
        file_sha256="f" * 64,
    )
    metrics = standardized_metrics(
        test_rows, [str(row["candidate_id"]) for row in selected], threshold
    )
    assert len(metrics["figure3"]) == 20
    assert metrics["table2"][0]["k"] == 10
    for name in ("figure3", "figure4", "table2", "prefix"):
        assert {row["method"] for row in metrics[name]} == {METHOD}


def test_test_bytes_are_opened_only_after_durable_selection_freeze(tmp_path: Path) -> None:
    test_csv = tmp_path / "test.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TASTEMOLNET_PREPARED_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "molecule_id": "sweet-test-1",
                "model_smiles": "CCO",
                "label": "1",
                "label_name": "Sweet",
                "split": "test",
                "exclusion_reason": "",
            }
        )
    raw = tmp_path / "raw"
    raw.mkdir()
    selection = raw / "selection_manifest.json"
    atomic_json(
        selection,
        {
            "schema_version": SELECTION_SCHEMA,
            "status": "FROZEN",
            "selection_frozen": True,
            "selector_fitted_on_calibration": True,
            "calibration_only": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
    )
    authority = SimpleNamespace(
        test_path=test_csv,
        declared_test_sha256=sha256_file(test_csv),
    )
    parents = _authorize_test(authority, selection)
    first_receipt = (raw / "test_access_receipt.json").read_bytes()
    assert [parent.parent_id for parent in parents] == ["sweet-test-1"]
    _authorize_test(authority, selection)
    assert (raw / "test_access_receipt.json").read_bytes() == first_receipt


def test_publisher_contract_and_verify_cli_requires_distinct_root() -> None:
    from scripts.verify_tastemolnet_gcf_full import build_parser

    assert STAGE == "T12_GCF_FULL"
    assert RUN_MANIFEST_SCHEMA == "tastemolnet_t12_final_run_manifest_v1"
    assert VERIFY_SCHEMA == "tastemolnet_t12_terminal_verification_v1"
    assert (PASS_MARKER + "\n").encode() == b"[TASTE_GCF_PASS]\n"
    required = {
        action.dest for action in build_parser()._actions if action.required
    }
    assert "verification_root" in required
    sidecar = (
        Path("scripts/autodl/run_tastemolnet_t12_paper_after_generation_v1.sh")
        .read_text(encoding="utf-8")
    )
    assert "fast16_matrix_cell_root_locator_v1" in sidecar
    assert "cell_root_locator.json" in sidecar
    assert "GENERATION_CONTROLLER_EXITED_WITHOUT_PASS" in sidecar


def test_generation_gate_rejects_nonexact_pass_bytes(tmp_path: Path) -> None:
    generation = tmp_path / "generation"
    verification = generation / "generation_verification"
    verification.mkdir(parents=True)
    (verification / "GENERATION_PASS").write_bytes(b"PASS\n")
    with pytest.raises(TasteGCFPostprocessError, match="PASS bytes"):
        _validate_generation_pass(generation, verification)


def test_distinct_terminal_verifier_publishes_exact_paper_pass_last(
    tmp_path: Path,
) -> None:
    output = tmp_path / "paper"
    verification = tmp_path / "verification"
    output.mkdir()
    verification.mkdir()
    audit = {
        "schema_version": VERIFY_SCHEMA,
        "status": "PASS",
        "passed": True,
        "audit_passed": True,
        "independent_verifier": True,
        "checks": {
            "calibration_only_selector_replayed": True,
            "selection_frozen_before_test": True,
        },
    }
    _publish_terminal_pass(
        output=output, verification=verification, audit=audit
    )
    assert (output / "PASS").read_bytes() == b"[TASTE_GCF_PASS]\n"
    assert (verification / "PASS").read_bytes() == b"[TASTE_GCF_PASS]\n"
    assert json.loads(
        (verification / "terminal_verification.json").read_text()
    )["schema_version"] == VERIFY_SCHEMA


def test_parent_chunks_resume_without_recomputing_distance(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    (output / "raw").mkdir(parents=True)
    parent = SimpleNamespace(
        parent_id="p0", smiles="parent-0", split="calibration", label=1
    )
    candidates = [_candidate(index) for index in range(10)]

    class _Scorer:
        def score_smiles(self, _values: list[str]) -> list[dict[str, object]]:
            return [{"predicted_label": 1, "probabilities": [0.1, 0.8, 0.1]}]

    first_rows, first_manifest = evaluate_split_resumable(
        split="calibration",
        parents=[parent],
        candidates=candidates,
        scorer=_Scorer(),
        provider=_Distance(),
        authority=_authority(),
        output=output,
        checkpoint_callback=lambda _count: None,
    )

    class _MustNotScore:
        def score_smiles(self, _values: list[str]) -> list[dict[str, object]]:
            raise AssertionError("resumed chunk unexpectedly rescored its parent")

    resumed_rows, resumed_manifest = evaluate_split_resumable(
        split="calibration",
        parents=[parent],
        candidates=candidates,
        scorer=_MustNotScore(),
        provider=_Distance(),
        authority=_authority(),
        output=output,
        checkpoint_callback=lambda _count: None,
    )
    assert resumed_rows == first_rows
    assert resumed_manifest == first_manifest
