from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from src.baselines.tastemolnet_comrecgc_full import (
    M_FALLBACK_MAX,
    M_MAX,
    GENERATION_PASS_MARKER,
    TasteComRecGCFullError,
    build_full_train_correct_source_cohort,
    fallback_checkpoint_targets,
    resource_cap_decision,
    validate_t14_full_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Row:
    molecule_id: str
    label: int = 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_full_cohort_is_all_and_only_train_correct_sweet_with_stable_order() -> None:
    graph_a = "a" * 64
    graph_b = "b" * 64
    rows = [_Row("z"), _Row("a"), _Row("m"), _Row("b")]
    selected, manifest, payload = build_full_train_correct_source_cohort(
        true_sweet_rows=rows,
        predictions=[1, 0, 1, 1],
        source_probabilities=[0.9, 0.2, 0.8, 0.7],
        canonical_graph_hashes=[graph_a, graph_b, graph_a, graph_b],
        train_csv_sha256="c" * 64,
        checkpoint_id="d" * 64,
    )

    assert [row["parent_id"] for row in selected] == ["b", "m", "z"]
    assert [row["canonical_graph_hash"] for row in selected].count(graph_a) == 2
    assert manifest["selection"] == "true_label == 1 and frozen_T3_GINE_prediction == 1"
    assert manifest["cohort_count"] == 3
    assert manifest["cohort_jsonl_sha256"] == _sha(payload)
    assert manifest["validation_loaded"] is False
    assert manifest["calibration_loaded"] is False
    assert manifest["test_loaded"] is False


def test_full_cohort_rejects_non_sweet_input_even_if_prediction_is_sweet() -> None:
    with pytest.raises(TasteComRecGCFullError, match="not true Sweet"):
        build_full_train_correct_source_cohort(
            true_sweet_rows=[_Row("x", label=0)],
            predictions=[1],
            source_probabilities=[0.9],
            canonical_graph_hashes=["a" * 64],
            train_csv_sha256="b" * 64,
            checkpoint_id="c" * 64,
        )


def test_resource_cap_uses_20k_then_one_25k_fallback() -> None:
    assert resource_cap_decision(
        completed_step=M_MAX, valid_unique_rule_count=10
    )["stop_reason"] == "RESOURCE_CAP_20K_VALID_UNIQUE_PASS"
    assert resource_cap_decision(
        completed_step=M_MAX, valid_unique_rule_count=9
    )["state"] == "EXTEND_ONCE_TO_25K"
    assert resource_cap_decision(
        completed_step=M_FALLBACK_MAX, valid_unique_rule_count=10
    )["stop_reason"] == "FALLBACK_CAP_25K_VALID_UNIQUE_PASS"
    assert resource_cap_decision(
        completed_step=M_FALLBACK_MAX, valid_unique_rule_count=9
    )["state"] == "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES"
    with pytest.raises(TasteComRecGCFullError, match="off cadence"):
        resource_cap_decision(completed_step=22_500, valid_unique_rule_count=99)
    assert fallback_checkpoint_targets(20_000) == (22_500, 25_000)
    assert fallback_checkpoint_targets(22_500) == (25_000,)
    with pytest.raises(TasteComRecGCFullError, match="cursor"):
        fallback_checkpoint_targets(17_500)


def test_independent_terminal_verifier_reopens_bounded_train_only_closure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "t14"
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    cohort = b'{"parent_id":"x"}\n'
    (root / "cohort.jsonl").write_bytes(cohort)
    cohort_manifest = {
        "status": "PASS",
        "policy": "FULL_TRAIN_CORRECT_SOURCE",
        "cohort_jsonl_sha256": _sha(cohort),
    }
    (root / "cohort_manifest.json").write_text(json.dumps(cohort_manifest))
    valid = {"valid_unique_rule_count": 10}
    resource = {"state": "STOP_AND_POSTPROCESS", "m_effective": M_MAX}
    (root / "valid_unique.json").write_text(json.dumps(valid))
    (root / "resource_cap_receipt.json").write_text(json.dumps(resource))
    checkpoint = checkpoint_root / f"checkpoint-{M_MAX:06d}.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_suffix(".json").write_text("{}")
    (root / "progress.json").write_text(
        json.dumps({"status": "PASS", "completed_step": M_MAX})
    )
    manifest = {
        "schema_version": "tastemolnet_t14_comrecgc_full_v1",
        "status": "PASS",
        "stage": "T14_COMRECGC_FULL",
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "calibration_status": "NOT_EVALUATED",
        "held_out_test_status": "NOT_EVALUATED",
        "export_status": "NOT_EVALUATED",
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "cohort_manifest_sha256": _sha((root / "cohort_manifest.json").read_bytes()),
        "cohort_jsonl_sha256": _sha(cohort),
        "resource_cap": resource,
        "valid_unique": valid,
    }
    (root / "generation_manifest.json").write_text(json.dumps(manifest))
    (root / "GENERATION_PASS").write_text(f"{GENERATION_PASS_MARKER}\n")

    receipt = validate_t14_full_output(root)
    assert receipt["status"] == "PASS"
    assert receipt["m_effective"] == M_MAX
    assert receipt["test_loaded"] is False
    assert receipt["method_cell_pass"] is False

    manifest["test_loaded"] = True
    (root / "generation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TasteComRecGCFullError, match="closure changed"):
        validate_t14_full_output(root)


def test_t14_launchers_keep_gpu1_budget_and_slurm_contract() -> None:
    autodl = (PROJECT_ROOT / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh").read_text()
    slurm = (PROJECT_ROOT / "scripts/slurm/run_tastemolnet_comrecgc_full.sh").read_text()
    for token in (
        "--gpu-index 1",
        "TASTEMOLNET_T14_OUTPUT",
        "RUN_GNN_ABLATION",
        "inference.fallback_to_heuristic=false",
    ):
        assert token in autodl
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
        assert token in slurm
