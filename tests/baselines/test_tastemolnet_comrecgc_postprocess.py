from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.globalgce_mutagenicity_adapter import TrainParent
from src.baselines.tastemolnet_comrecgc_postprocess import (
    DATASET,
    METHOD,
    PASS_MARKER,
    PostprocessAuthority,
    TasteComRecGCPostprocessError,
    atomic_csv,
    atomic_json,
    compute_t14_standardized_metrics,
    evaluate_one_parent,
    evaluate_split_resumable,
    materialize_generation_candidates,
    select_candidates_on_calibration,
    stable_sha256,
    verify_pair_chunk_inventory,
)
from src.baselines.tastemolnet_globalgce_full import ThresholdContract
from src.eval.four_by_four_registry import PASS_STATUSES, audit_explicit_candidate


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _candidate(index: int) -> dict[str, object]:
    graph_hash = _sha(index + 1)
    return {
        "dataset": DATASET,
        "method": METHOD,
        "stage": "T14_COMRECGC_FULL_POSTPROCESS",
        "candidate_id": f"comrecgc_{graph_hash}",
        "candidate_content_hash": graph_hash,
        "generation_rank": index + 1,
        "cluster_id": index,
        "canonical_smiles": "CC" if index % 2 == 0 else "CO",
        "destination_label": 0 if index % 2 == 0 else 2,
        "score": 0.8,
        "frequency": 3,
        "covered_parent_count": 2,
        "cluster_size": 3,
        "lineage_count": 1,
        "lineage_sha256": _sha(1000 + index),
        "source_split": "train",
        "action_kind": "full_graph_common_recourse",
        "action_semantics": "official_comrecgc_cluster_representative_v1",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
    }


def _generation_payload(count: int = 10) -> tuple[dict, dict]:
    common_rows = []
    records = {}
    lineages = {}
    for index in range(count):
        graph_hash = _sha(index + 1)
        destination = 0 if index % 2 == 0 else 2
        common_rows.append(
            {
                "rank": index + 1,
                "cluster_id": index,
                "representative_graph_identity_sha256": graph_hash,
                "destination_label": destination,
                "score": 0.8,
                "frequency": 3,
                "covered_parent_count": 2,
                "cluster_size": 3,
                "lineage_count": 1,
            }
        )
        records[graph_hash] = {
            "graph_identity_sha256": graph_hash,
            "canonical_graph": "CC" if index % 2 == 0 else "CO",
            "prediction": destination,
            "candidate": True,
            "valid_fullgraph": True,
        }
        lineages[graph_hash] = {_sha(100 + index): 2}
    return (
        {
            "common_recourse": {
                "selected_common_recourse_count": count,
                "selected_common_recourses": common_rows,
            }
        },
        {"records": records, "lineage_occurrences": lineages},
    )


def _threshold(tmp_path: Path) -> ThresholdContract:
    values = (0.1, 0.2, 0.3)
    return ThresholdContract(
        values=values,
        theta_star=0.2,
        cost_cap=0.3,
        config_hash=stable_sha256(list(values)),
        source="unit-calibration",
        source_split="calibration",
        file_sha256=_sha(901),
    )


def _authority(tmp_path: Path) -> PostprocessAuthority:
    return PostprocessAuthority(
        generation_root=tmp_path / "generation",
        calibration_path=tmp_path / "calibration.csv",
        test_path=tmp_path / "test.csv",
        checkpoint_path=tmp_path / "checkpoint",
        molclr_root=tmp_path / "molclr",
        molclr_checkpoint=tmp_path / "molclr.pt",
        threshold_path=tmp_path / "threshold.json",
        generation_inventory_sha256=_sha(1),
        generation_manifest_sha256=_sha(2),
        generation_checkpoint_digest=_sha(3),
        generation_effective_m=20_000,
        generation_stop_reason="RESOURCE_CAP_20K_VALID_UNIQUE_PASS",
        generation_resource_cap_used=True,
        generation_early_stop_used=False,
        calibration_sha256=_sha(4),
        declared_test_sha256=_sha(5),
        checkpoint_id=_sha(6),
        temperature_calibration_hash=_sha(60),
        dataset_hash=_sha(7),
        split_manifest_sha256=_sha(8),
        molclr_checkpoint_sha256=_sha(9),
        threshold=_threshold(tmp_path),
    )


class _FakeScorer:
    checkpoint_id = _sha(6)

    def score_smiles(self, values: list[str]) -> list[dict]:
        rows = []
        for value in values:
            if value.startswith("P"):
                prediction = 1
                probabilities = [0.1, 0.8, 0.1]
            elif value == "CC":
                prediction = 0
                probabilities = [0.8, 0.1, 0.1]
            else:
                prediction = 2
                probabilities = [0.1, 0.1, 0.8]
            rows.append(
                {
                    "checkpoint_id": self.checkpoint_id,
                    "num_classes": 3,
                    "source_label": 1,
                    "backbone": "gine",
                    "predicted_label": prediction,
                    "probabilities": probabilities,
                }
            )
        return rows


class _FakeProvider:
    def distance(self, parent: str, candidate: str) -> dict:
        return {"ok": True, "distance": 0.1 if candidate == "CC" else 0.25}


def test_materializes_only_official_checkpointed_common_recourses() -> None:
    pytest.importorskip("rdkit")
    generation, bridge = _generation_payload()
    rows = materialize_generation_candidates(
        generation_manifest=generation, bridge_state=bridge
    )
    assert len(rows) == 10
    assert [row["generation_rank"] for row in rows] == list(range(1, 11))
    assert {row["destination_label"] for row in rows} == {0, 2}
    assert all(row["source_split"] == "train" for row in rows)
    assert all(row["lineage_count"] == 1 for row in rows)


def test_materialization_rejects_fewer_than_ten() -> None:
    generation, bridge = _generation_payload(9)
    with pytest.raises(TasteComRecGCPostprocessError, match="10..20"):
        materialize_generation_candidates(
            generation_manifest=generation, bridge_state=bridge
        )


def test_evaluates_multiclass_fullgraph_candidate_without_binary_collapse() -> None:
    rows = evaluate_one_parent(
        parent=TrainParent("p0", "PARENT", 1, "calibration"),
        candidates=[_candidate(0), _candidate(1)],
        scorer=_FakeScorer(),
        provider=_FakeProvider(),
        split="calibration",
    )
    assert [row["destination_label"] for row in rows] == [0, 2]
    assert all(row["pair_strict_flip"] is True for row in rows)
    assert all(row["pred_before"] == 1 for row in rows)
    assert all(row["rf_oracle_used"] is False for row in rows)


def test_parent_chunks_resume_and_reject_tampering(tmp_path: Path) -> None:
    output = tmp_path / "science"
    (output / "raw").mkdir(parents=True)
    parents = [
        TrainParent("p0", "PARENT0", 1, "calibration"),
        TrainParent("p1", "PARENT1", 1, "calibration"),
    ]
    candidates = [_candidate(index) for index in range(10)]
    authority = _authority(tmp_path)
    rows, manifest = evaluate_split_resumable(
        split="calibration",
        parents=parents,
        candidates=candidates,
        scorer=_FakeScorer(),
        provider=_FakeProvider(),
        output=output,
        authority=authority,
        resume_identity=authority.resume_identity(),
    )
    assert len(rows) == 20
    assert manifest["resumable_parent_chunks"] is True
    inventory = verify_pair_chunk_inventory(
        output=output, split="calibration", pair_rows=rows
    )
    assert inventory["pair_count"] == 20
    resumed, _ = evaluate_split_resumable(
        split="calibration",
        parents=parents,
        candidates=candidates,
        scorer=_FakeScorer(),
        provider=_FakeProvider(),
        output=output,
        authority=authority,
        resume_identity=authority.resume_identity(),
    )
    assert resumed == rows
    chunk = output / "raw/calibration_pair_chunks/00000000.jsonl"
    with chunk.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tamper": True}) + "\n")
    with pytest.raises(TasteComRecGCPostprocessError, match="resume chunk changed"):
        evaluate_split_resumable(
            split="calibration",
            parents=parents,
            candidates=candidates,
            scorer=_FakeScorer(),
            provider=_FakeProvider(),
            output=output,
            authority=authority,
            resume_identity=authority.resume_identity(),
        )


def _pair_rows(split: str) -> tuple[list[dict], list[dict]]:
    candidates = [_candidate(index) for index in range(10)]
    rows = []
    for parent_index in range(3):
        for index, candidate in enumerate(candidates):
            distance = 0.1 + 0.01 * index
            rows.append(
                {
                    "dataset": DATASET,
                    "method": METHOD,
                    "split": split,
                    "parent_id": f"p{parent_index}",
                    "candidate_id": candidate["candidate_id"],
                    "pair_strict_flip": True,
                    "wnode_distance": distance,
                    "destination_label": candidate["destination_label"],
                    "pred_before": 1,
                    "pred_after": candidate["destination_label"],
                    "cf_drop": 0.7,
                    "applicable": True,
                    "rf_oracle_used": False,
                }
            )
    return candidates, rows


def test_calibration_order_and_standardized_exports_are_comrecgc(tmp_path: Path) -> None:
    candidates, calibration = _pair_rows("calibration")
    selected, selection = select_candidates_on_calibration(
        candidates, calibration, theta_star=0.2
    )
    assert len(selected) == 10
    assert selection["selector_fitted_on_calibration"] is True
    assert selection["test_loaded"] is False
    assert "ordered_rule_ids" not in selection
    _unused, test_rows = _pair_rows("test")
    metrics = compute_t14_standardized_metrics(
        test_rows,
        [str(row["candidate_id"]) for row in selected],
        _threshold(tmp_path),
    )
    assert len(metrics["figure3"]) == 20
    assert len(metrics["table2"]) == 1
    assert all(row["method"] == METHOD for row in metrics["figure3"])
    assert all(row["method"] == METHOD for row in metrics["destination"])


def test_terminal_marker_is_matrix_publisher_contract() -> None:
    assert PASS_MARKER == "[TASTE_COMRECGC_PASS]"


def test_final_manifest_shape_is_four_by_four_registry_eligible(tmp_path: Path) -> None:
    root = tmp_path / "final"
    raw = tmp_path / "science"
    oracle = tmp_path / "checkpoint"
    root.mkdir()
    raw.mkdir()
    oracle.mkdir()
    (raw / "raw.txt").write_text("closed\n", encoding="utf-8")
    threshold = _threshold(tmp_path)
    _candidates, test_rows = _pair_rows("test")
    metrics = compute_t14_standardized_metrics(
        test_rows,
        [str(_candidate(index)["candidate_id"]) for index in range(10)],
        threshold,
    )
    for name, rows in {
        "figure3_coverage_vs_k.csv": metrics["figure3"],
        "figure4_coverage_vs_threshold.csv": metrics["figure4"],
        "prefix_metrics.csv": metrics["prefix"],
        "parent_best_distances.csv": metrics["parent_best"],
        "destination_distribution.csv": metrics["destination"],
        "table2_comrecgc_k10.csv": metrics["table2"],
    }.items():
        atomic_csv(root / name, rows)
    atomic_json(root / "prefix_metrics.json", metrics["prefix"])
    common = {
        "dataset": DATASET,
        "method": METHOD,
        "stage": "T14_COMRECGC_FULL_POSTPROCESS",
        "num_classes": 3,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(oracle),
        "oracle_hash": _sha(6),
        "oracle_checkpoint_hash": _sha(6),
        "temperature_calibration_hash": _sha(60),
        "dataset_hash": _sha(7),
        "test_parent_ids_sha256": _sha(70),
        "test_split_hash": _sha(5),
        "distance_line": "MolCLR-Node-Wasserstein",
        "molclr_checkpoint_hash": _sha(9),
        "cf_mode": "strict_flip",
        "threshold_config_hash": threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(raw),
        "source_generation_root": str(raw),
        "generation_adopted": True,
        "M_configured_max": 20_000,
        "M_fallback_max": 25_000,
        "M_effective": 20_000,
        "resource_cap_used": True,
        "early_stop_used": False,
        "stop_reason": "RESOURCE_CAP_20K_VALID_UNIQUE_PASS",
    }
    atomic_json(
        root / "summary.json",
        {
            "schema_version": "tastemolnet_t14_summary_v1",
            **common,
            "status": "PASS",
            "frozen": True,
            "artifacts_frozen": True,
            "raw_output_complete": True,
            "raw_artifacts_complete": True,
        },
    )
    atomic_json(
        root / "run_manifest.json",
        {
            "schema_version": "tastemolnet_t14_postprocess_run_manifest_v1",
            **common,
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "raw_output_complete": True,
            "raw_artifacts_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
            "artifacts_frozen": True,
            "selection_frozen_before_test": True,
            "independent_terminal_verification_passed": True,
        },
    )
    atomic_json(
        root / "final_artifact_audit.json",
        {
            "schema_version": "tastemolnet_t14_postprocess_terminal_verification_v1",
            **common,
            "status": "PASS",
            "passed": True,
            "audit_passed": True,
            "independent_verifier": True,
            "raw_output_complete": True,
            "raw_artifacts_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
            "artifacts_frozen": True,
        },
    )
    atomic_json(
        root / "oracle_manifest.json",
        {
            "schema_version": "tastemolnet_t14_oracle_manifest_v1",
            **common,
            "same_frozen_gine_for_generation_calibration_test": True,
            "frozen": True,
        },
    )
    atomic_json(
        root / "evaluation_manifest.json",
        {
            "schema_version": "tastemolnet_t14_evaluation_manifest_v1",
            **common,
            "selection_frozen_before_test": True,
            "full_cartesian_test_pairs": True,
            "frozen": True,
        },
    )
    result = audit_explicit_candidate(root, dataset=DATASET, method=METHOD)
    assert result.status in PASS_STATUSES, result.reason_codes
