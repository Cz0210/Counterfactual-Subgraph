from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluate_ccrcov_with_molclr_node_wasserstein import (
    validate_preselected_candidate_csv,
)
from src.eval.close_counterfactual_coverage import _load_candidate_records
from src.eval.fullgraph_wnode_artifacts import (
    finalize_fullgraph_evaluation_run,
    load_frozen_threshold_contract,
    load_ranked_candidates,
    stable_json_sha256,
    summarize_wnode_thresholds,
    validate_frozen_candidate_contract,
)
from src.eval.greed_distance.pair_generation import GT_FULLGRAPH_FIELDS


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_WRAPPER = (
    ROOT
    / "scripts/slurm/gcfexplainer/build_mutagenicity_wnode_calibration_matrix.sh"
)
TEST_WRAPPER = (
    ROOT
    / "scripts/slurm/gcfexplainer/evaluate_mutagenicity_wnode_frozen_test.sh"
)
NATIVE_RANKS = (
    112,
    120,
    177,
    179,
    195,
    217,
    388,
    442,
    605,
    701,
    794,
    810,
    1034,
    1095,
    1417,
    1786,
    1788,
    1975,
    3815,
    4198,
)
SMILES = (
    "C",
    "N",
    "O",
    "F",
    "Cl",
    "Br",
    "CC",
    "CN",
    "CO",
    "C=C",
    "C#N",
    "CCC",
    "CCN",
    "CCO",
    "CCF",
    "CCCl",
    "CCBr",
    "CNC",
    "COC",
    "N#N",
)
SELECTION_METHOD = "native_gcf_summary_rank_filtered_by_validity"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_inputs(root: Path) -> tuple[Path, Path, str, str]:
    rows = [
        {
            "candidate_id": f"gcf-{index:02d}",
            "native_rank": native_rank,
            "smiles": smiles,
            "canonical_smiles": smiles,
            "rdkit_valid": True,
            "rf_pred": 0,
            "rf_prob_0": 0.9,
            "rf_prob_1": 0.1,
            "source_method": "GCFExplainer",
            "selection_method": SELECTION_METHOD,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "projected_new_edge_count": 0,
            "retained_edge_count": 1,
            "source_parent_id": "train-parent",
        }
        for index, (native_rank, smiles) in enumerate(
            zip(NATIVE_RANKS, SMILES), start=1
        )
    ]
    candidates = root / "export/selected_top20.csv"
    _write_csv(candidates, rows)
    csv_sha = hashlib.sha256(candidates.read_bytes()).hexdigest()
    order_sha = stable_json_sha256([row["candidate_id"] for row in rows])
    manifest = root / "frozen_candidate_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "Mutagenicity",
                "source_label": 1,
                "target_label": 0,
                "candidate_count": 20,
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": SELECTION_METHOD,
                "rf_reranking_performed": False,
                "wnode_reranking_performed": False,
                "candidate_csv_sha256": csv_sha,
                "selected_candidate_order_sha256": order_sha,
                "selected_native_ranks": list(NATIVE_RANKS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return candidates, manifest, csv_sha, order_sha


def _thresholds(path: Path) -> tuple[float, ...]:
    values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
    path.write_text(
        json.dumps(
            {
                "threshold_source": "frozen_calibration",
                "theta_star": 0.04,
                "cost_cap": 0.1,
                "raw_quantile_thresholds": [
                    {"threshold": value} for value in values
                ],
            }
        ),
        encoding="utf-8",
    )
    return values


def test_existing_fullgraph_loader_reads_frozen_gcf_csv_without_adapter(
    tmp_path: Path,
) -> None:
    candidates_path, manifest_path, csv_sha, order_sha = _candidate_inputs(
        tmp_path
    )
    validation = validate_preselected_candidate_csv(candidates_path, 20)
    _, loaded = _load_candidate_records(
        candidates_path,
        fields=GT_FULLGRAPH_FIELDS,
        directory_candidates=(),
    )
    contract = validate_frozen_candidate_contract(
        candidates_csv=candidates_path,
        frozen_manifest_path=manifest_path,
        expected_count=20,
        expected_csv_sha256=csv_sha,
        expected_order_sha256=order_sha,
        expected_native_ranks=NATIVE_RANKS,
        expected_selection_method=SELECTION_METHOD,
    )

    assert validation["num_rows"] == 20
    assert [item.candidate_id for item in loaded] == [
        f"gcf-{index:02d}" for index in range(1, 21)
    ]
    assert [int(item.raw["native_rank"]) for item in loaded] == list(NATIVE_RANKS)
    assert contract["adapter_used"] is False
    assert contract["candidate_csv_sha256"] == csv_sha
    assert contract["selected_candidate_order_sha256"] == order_sha
    assert contract["candidate_set_preselected"] is True
    assert contract["selection_performed_in_eval"] is False
    assert not (ROOT / "src/baselines/gcfexplainer_wnode_adapter.py").exists()


def test_native_rank_loader_uses_csv_order_as_frozen_prefix(tmp_path: Path) -> None:
    candidates_path, _, _, _ = _candidate_inputs(tmp_path)
    candidates, fields = load_ranked_candidates(candidates_path, expected_count=20)
    assert "native_rank" in fields
    assert [row["rank"] for row in candidates] == list(range(1, 21))
    assert [int(row["native_rank"]) for row in candidates] == list(NATIVE_RANKS)

    rows = list(csv.DictReader(candidates_path.open(encoding="utf-8")))
    rows[1], rows[2] = rows[2], rows[1]
    _write_csv(candidates_path, rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_ranked_candidates(candidates_path, expected_count=20)


def test_candidate_contract_rejects_hash_or_order_changes(tmp_path: Path) -> None:
    candidates_path, manifest_path, csv_sha, order_sha = _candidate_inputs(tmp_path)
    with pytest.raises(ValueError, match="CSV SHA256 mismatch"):
        validate_frozen_candidate_contract(
            candidates_csv=candidates_path,
            frozen_manifest_path=manifest_path,
            expected_count=20,
            expected_csv_sha256="0" * 64,
            expected_order_sha256=order_sha,
            expected_native_ranks=NATIVE_RANKS,
            expected_selection_method=SELECTION_METHOD,
        )
    with pytest.raises(ValueError, match="order SHA256 mismatch"):
        validate_frozen_candidate_contract(
            candidates_csv=candidates_path,
            frozen_manifest_path=manifest_path,
            expected_count=20,
            expected_csv_sha256=csv_sha,
            expected_order_sha256="0" * 64,
            expected_native_ranks=NATIVE_RANKS,
            expected_selection_method=SELECTION_METHOD,
        )


@pytest.mark.parametrize("cohort", ("calibration", "test"))
def test_cartesian_run_audit_preserves_frozen_order_and_selection_semantics(
    tmp_path: Path,
    cohort: str,
) -> None:
    candidates_path, manifest_path, csv_sha, order_sha = _candidate_inputs(
        tmp_path / "frozen"
    )
    thresholds_path = tmp_path / "thresholds.json"
    thresholds = _thresholds(thresholds_path)
    dataset_path = tmp_path / f"{cohort}_source_label1_teacher_correct.csv"
    _write_csv(
        dataset_path,
        [
            {"molecule_id": "p1", "smiles": "CCO", "label": 1},
            {"molecule_id": "p2", "smiles": "CCN", "label": 1},
        ],
    )
    run_dir = tmp_path / f"{cohort}_run"
    (run_dir / "details").mkdir(parents=True)
    (run_dir / "combined").mkdir()
    details = [
        {
            "method": "GCFExplainer-Top20",
            "parent_id": parent_id,
            "parent_smiles": "CCO",
            "label": 1,
            "candidate_id": f"gcf-{candidate_index:02d}",
            "candidate_smiles": SMILES[candidate_index - 1],
            "match": True,
            "delete_valid": True,
            "pred_before": 1,
            "pred_after": 0,
            "cf_flip": True,
            "teacher_strict_flip": True,
            "cf_drop": 0.8,
            "distance": 0.001 * candidate_index,
        }
        for parent_id in ("0", "1")
        for candidate_index in range(1, 21)
    ]
    _write_csv(run_dir / "details/pair_details.csv", details)
    summary = summarize_wnode_thresholds(
        method="GCFExplainer-Top20",
        details=details,
        threshold_rows=[
            {"threshold": value, "threshold_source": "explicit", "quantile": None}
            for value in thresholds
        ],
        total_parents=2,
        total_candidates=20,
        source_label=1,
        target_label=0,
    )
    _write_csv(run_dir / "combined/combined_threshold_summary.csv", summary)
    teacher_path = tmp_path / "teacher.pkl"
    teacher_path.write_bytes(b"teacher")
    molclr_checkpoint = tmp_path / "model.pth"
    molclr_checkpoint.write_bytes(b"molclr")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_csv": str(dataset_path),
                "teacher_path": str(teacher_path),
                "molclr_checkpoint": str(molclr_checkpoint),
                "label": 1,
                "label_col": "label",
                "max_parents": 2,
                "max_candidates": 20,
                "preselected_topk": 20,
                "threshold_source": "explicit",
                "thresholds": list(thresholds),
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_method": SELECTION_METHOD,
                "run_ours": False,
                "run_fullgraph": True,
                "cf_mode": "strict_flip",
                "feature_cost": "cosine",
                "node_mass": "uniform",
                "size_penalty_beta": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "_EVALUATOR_COMPLETE.json").write_text(
        json.dumps(
            {
                "complete": True,
                "num_unique_parent_candidate_pairs": 40,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = finalize_fullgraph_evaluation_run(
        run_dir=run_dir,
        frozen_candidates_csv=candidates_path,
        frozen_manifest_path=manifest_path,
        thresholds_json=thresholds_path,
        cohort_name=cohort,
        expected_parent_count=2,
        expected_candidate_count=20,
        expected_pair_count=40,
        expected_candidate_csv_sha256=csv_sha,
        expected_candidate_order_sha256=order_sha,
        expected_teacher_sha256=hashlib.sha256(
            teacher_path.read_bytes()
        ).hexdigest(),
        expected_molclr_checkpoint_sha256=hashlib.sha256(
            molclr_checkpoint.read_bytes()
        ).hexdigest(),
        expected_native_ranks=NATIVE_RANKS,
        expected_method="GCFExplainer-Top20",
        expected_selection_method=SELECTION_METHOD,
    )
    assert audit["pair_count"] == 40
    assert audit["complete_cartesian"] is True
    assert audit["all_pair_distances_finite"] is True
    assert audit["all_source_teacher_predictions_match"] is True
    assert audit["all_candidate_teacher_predictions_match"] is True
    assert audit["all_pairs_strict_flip"] is True
    assert audit["candidate_selection_performed"] is False
    assert audit["selection_used_calibration"] is False
    assert audit["selection_used_test"] is False
    assert audit["threshold_fitted_on_test"] is False
    assert audit["threshold_provenance"]["thresholds_json_sha256"] == hashlib.sha256(
        thresholds_path.read_bytes()
    ).hexdigest()
    assert (run_dir / "_RUN_COMPLETE.json").is_file()


def test_frozen_threshold_contract_never_uses_auto_quantile(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    expected = _thresholds(path)
    contract = load_frozen_threshold_contract(path)
    assert tuple(contract["thresholds"]) == expected
    assert contract["threshold_source"] == "frozen_calibration"


def test_gcf_wnode_wrappers_are_safe_fullgraph_only_entrypoints() -> None:
    for wrapper in (CALIBRATION_WRAPPER, TEST_WRAPPER):
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        text = wrapper.read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "#SBATCH --cpus-per-task=7" in text
        assert "unset http_proxy" not in text
        assert "unset https_proxy" not in text
        assert "select_mutagenicity_wnode_prefix" not in text
        assert "auto_quantile" not in text
        assert '--run-ours 0' in text
        assert '--run-fullgraph 1' in text
        assert '--cf-mode strict_flip' in text
        assert '--fullgraph-candidates-path "$FULLGRAPH_CANDIDATES_PATH"' in text
        assert 'RESUME="${RESUME:-false}"' in text
        assert "native_gcf_summary_rank_filtered_by_validity" in text
        assert "load_frozen_threshold_contract" in text
        assert "0.038576244576299636" not in text

    calibration = CALIBRATION_WRAPPER.read_text(encoding="utf-8")
    frozen_test = TEST_WRAPPER.read_text(encoding="utf-8")
    assert "EXPECTED_PARENT_COUNT=235" in calibration
    assert "EXPECTED_PAIR_COUNT=4700" in calibration
    assert "calibration_source_label1_teacher_correct.csv" in calibration
    assert "test_source" not in calibration
    assert "EXPECTED_PARENT_COUNT=217" in frozen_test
    assert "EXPECTED_PAIR_COUNT=4340" in frozen_test
    assert "test_source_label1_teacher_correct.csv" in frozen_test
    assert "export_wnode_final_artifacts.py" in frozen_test
    assert "audit_wnode_final_artifacts.py" in frozen_test
    assert "table2_gcfexplainer_k10.csv" in frozen_test
    assert "table2_gcfexplainer_k20.csv" in frozen_test


def test_official_gcf_core_is_not_modified_by_wnode_integration() -> None:
    official = ROOT / "baselines/gcfexplainer_official"
    for path in official.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mutagenicity_gcfexplainer_wnode_v1" not in text
        assert "selected_candidate_order_sha256" not in text
