from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.mutagenicity_wnode_frozen_test import (
    FrozenTestConfig,
    FrozenTestInterrupted,
    audit_frozen_test_run,
    build_frozen_test_run,
    compute_frozen_prefix_metrics,
    frozen_threshold_output,
    load_and_verify_frozen_selector,
)
from src.eval.mutagenicity_wnode_matrix import (
    CalibrationParent,
    evaluate_parent_candidate_pair,
)


FRAGMENTS = (
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


class FakeTeacher:
    def score_smiles(self, smiles, label=None, **kwargs):
        del kwargs
        parent = smiles in {"CCCO", "CCCN"}
        probability = 0.9 if parent else 0.2
        prediction = 1 if parent else 0
        return {
            "teacher_result_ok": True,
            "teacher_reason": "ok",
            "teacher_label": prediction,
            "teacher_prob": probability if label == 1 else 1.0 - probability,
        }


class FakeDistance:
    def __init__(self):
        self.calls = []

    def distance(self, left, right):
        self.calls.append((left, right))
        if left == right:
            value = 0.0
        elif {left, right} == {"CCO", "CCN"}:
            value = 0.1
        elif right == "CN":
            value = 0.02
        elif right == "CC":
            value = 0.03
        else:
            value = 0.05
        return {
            "distance": value,
            "ok": True,
            "cache_hit": False,
            "error": None,
        }

    def stats_dict(self):
        return {
            "pair_distance_cache_hit_rate": 0.0,
            "node_embedding_cache_hit_rate": 0.0,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _thresholds():
    quantiles = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
    weights = (4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0)
    values = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    labels = ("q05", "q10", "q20", "q30", "q50", "q70", "q90")
    return {
        "finite_strict_flip_distance_count": 1573,
        "quantile_method": "linear",
        "dtype": "float64",
        "requested_quantiles": list(quantiles),
        "requested_weights": list(weights),
        "raw_quantile_thresholds": [
            {
                "quantile": quantile,
                "quantile_label": label,
                "threshold": threshold,
                "weight": weight,
            }
            for quantile, label, threshold, weight in zip(
                quantiles,
                labels,
                values,
                weights,
            )
        ],
        "merged_thresholds": [
            {
                "threshold_id": label,
                "threshold": threshold,
                "weight": weight,
                "quantiles": [quantile],
                "quantile_labels": [label],
            }
            for quantile, label, threshold, weight in zip(
                quantiles,
                labels,
                values,
                weights,
            )
        ],
        "duplicate_thresholds_merged": False,
        "theta_star_quantile": 0.30,
        "theta_star": 0.04,
        "cost_cap_quantile": 0.90,
        "cost_cap": 0.07,
        "threshold_source": "calibration_all_finite_strict_flip_pairs",
        "test_used": False,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_frozen_package(root: Path) -> Path:
    selected_dir = root / "selected_variant"
    selected_dir.mkdir(parents=True)
    rows = [
        {
            "rank": index,
            "candidate_id": f"C{index:02d}",
            "canonical_fragment": fragment,
            "source_parent_count": 100 - index,
            "source_cf_drop_mean": 0.5,
            "source_reward_mean": 1.0,
        }
        for index, fragment in enumerate(FRAGMENTS, start=1)
    ]
    (selected_dir / "selected_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidate_ids = [row["candidate_id"] for row in rows]
    _write_json(
        selected_dir / "selected_top10.json",
        {"candidate_ids": candidate_ids[:10], "candidates": rows[:10]},
    )
    _write_json(
        selected_dir / "selected_top20.json",
        {"candidate_ids": candidate_ids, "candidates": rows},
    )
    common = {
        "frozen": True,
        "selected_variant": "A2_MultiThreshold",
        "top_k": 20,
        "table_k": 10,
        "test_used_for_selection": False,
    }
    _write_json(root / "_FROZEN.json", common)
    _write_json(
        root / "calibration_decision.json",
        {
            "selected_variant": "A2_MultiThreshold",
            "test_used_for_selection": False,
        },
    )
    _write_json(root / "thresholds.json", _thresholds())
    required = (
        "_FROZEN.json",
        "thresholds.json",
        "calibration_decision.json",
        "selected_variant/selected_sequence.jsonl",
        "selected_variant/selected_top10.json",
        "selected_variant/selected_top20.json",
    )
    manifest = {
        **common,
        "file_sha256": {
            relative: _sha256(root / relative) for relative in required
        },
    }
    _write_json(root / "frozen_selector_manifest.json", manifest)
    return root


def _make_test_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecule_id", "smiles", "label", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "molecule_id": "P1",
                "smiles": "CCCO",
                "label": 1,
                "split": "test",
            }
        )
        writer.writerow(
            {
                "molecule_id": "P2",
                "smiles": "CCCN",
                "label": 1,
                "split": "test",
            }
        )
    return path


def _deletions(parent, fragment):
    del parent
    if fragment != "C":
        return []
    return [
        {
            "match_index": 0,
            "match_atoms": [0],
            "delete_valid": True,
            "residual_smiles": "CC",
            "error": None,
        },
        {
            "match_index": 1,
            "match_atoms": [1],
            "delete_valid": True,
            "residual_smiles": "CN",
            "error": None,
        },
    ]


def _run_inputs(tmp_path):
    frozen = _make_frozen_package(tmp_path / "frozen")
    test_csv = _make_test_csv(tmp_path / "fake_final_test.csv")
    teacher_path = tmp_path / "teacher.pkl"
    teacher_path.write_bytes(b"teacher")
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    checkpoint = tmp_path / "molclr.pt"
    checkpoint.write_bytes(b"checkpoint")
    return frozen, test_csv, teacher_path, molclr_root, checkpoint


def test_frozen_hashes_candidate_order_and_fixed_thresholds(tmp_path):
    frozen = _make_frozen_package(tmp_path / "frozen")
    package = load_and_verify_frozen_selector(frozen)
    assert package.selected_variant == "A2_MultiThreshold"
    assert package.candidate_ids == tuple(f"C{index:02d}" for index in range(1, 21))
    assert package.thresholds.theta_star == pytest.approx(0.04)
    assert package.thresholds.cost_cap == pytest.approx(0.07)
    output = frozen_threshold_output(package)
    assert output["threshold_source"] == "frozen_calibration_selector"
    assert output["test_threshold_fitting"] is False
    assert output["test_candidate_selection"] is False
    assert output["test_variant_selection"] is False

    with (frozen / "thresholds.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_and_verify_frozen_selector(frozen)


def test_multiple_matches_use_minimum_strict_flip_wnode():
    pair, matches = evaluate_parent_candidate_pair(
        CalibrationParent("P1", "CCCO", 1, "test"),
        {"candidate_id": "C01", "canonical_fragment": "C"},
        teacher=FakeTeacher(),
        distance_provider=FakeDistance(),
        deletion_fn=_deletions,
    )
    assert len(matches) == 2
    assert pair["pair_strict_flip"] is True
    assert pair["best_match_index"] == 1
    assert pair["wnode_distance"] == pytest.approx(0.02)
    assert pair["residual_smiles"] == "CN"
    assert all(row["teacher_strict_flip"] for row in matches)


def test_frozen_cartesian_prefix_artifacts_and_independent_audit(tmp_path):
    frozen, test_csv, teacher_path, molclr_root, checkpoint = _run_inputs(tmp_path)
    output = tmp_path / "output"
    summary = build_frozen_test_run(
        frozen_selector_root=frozen,
        test_csv=test_csv,
        teacher_path=teacher_path,
        molclr_root=molclr_root,
        molclr_checkpoint=checkpoint,
        output_dir=output,
        wnode_cache_db=tmp_path / "cache.sqlite",
        teacher=FakeTeacher(),
        distance_provider=FakeDistance(),
        config=FrozenTestConfig(expected_parent_count=2, flush_every=3),
        deletion_fn=_deletions,
    )
    assert summary["actual_pair_rows"] == 40
    assert summary["complete_cartesian"] is True
    assert summary["test_threshold_fitting"] is False
    pairs = [
        json.loads(line)
        for line in (output / "pair_matrix.jsonl").read_text().splitlines()
    ]
    assert len(pairs) == 40
    assert len({(row["parent_id"], row["candidate_id"]) for row in pairs}) == 40
    inapplicable = [row for row in pairs if row["candidate_id"] != "C01"]
    assert len(inapplicable) == 38
    assert all(row["applicable"] is False for row in inapplicable)
    assert all(row["wnode_distance"] is None for row in inapplicable)

    prefix = list(csv.DictReader((output / "prefix_metrics.csv").open()))
    assert [int(row["k"]) for row in prefix] == list(range(1, 21))
    coverage = np.asarray([float(row["ccrcov_theta_star"]) for row in prefix])
    capped = np.asarray([float(row["fixed_capped_mean_cost"]) for row in prefix])
    assert np.all(np.diff(coverage) >= -1e-12)
    assert np.all(np.diff(capped) <= 1e-12)
    assert (output / "table2_ours_k10.csv").is_file()
    assert (output / "table2_ours_k20.csv").is_file()
    assert (output / "figure3_coverage_vs_k.csv").is_file()
    assert (output / "figure4_coverage_vs_threshold.csv").is_file()

    audit = audit_frozen_test_run(
        output,
        frozen_selector_root=frozen,
        test_csv=test_csv,
        expected_parent_count=2,
        expected_candidate_count=20,
        expected_pair_count=40,
    )
    assert audit["audit_passed"] is True
    assert audit["complete_cartesian"] is True

    with pytest.raises(FileExistsError, match="cannot be rerun"):
        build_frozen_test_run(
            frozen_selector_root=frozen,
            test_csv=test_csv,
            teacher_path=teacher_path,
            molclr_root=molclr_root,
            molclr_checkpoint=checkpoint,
            output_dir=output,
            wnode_cache_db=tmp_path / "cache.sqlite",
            teacher=FakeTeacher(),
            distance_provider=FakeDistance(),
            config=FrozenTestConfig(expected_parent_count=2),
            deletion_fn=_deletions,
        )


def test_resume_does_not_duplicate_or_omit_pairs(tmp_path):
    frozen, test_csv, teacher_path, molclr_root, checkpoint = _run_inputs(tmp_path)
    output = tmp_path / "resume_output"
    kwargs = {
        "frozen_selector_root": frozen,
        "test_csv": test_csv,
        "teacher_path": teacher_path,
        "molclr_root": molclr_root,
        "molclr_checkpoint": checkpoint,
        "output_dir": output,
        "wnode_cache_db": tmp_path / "cache.sqlite",
        "teacher": FakeTeacher(),
        "distance_provider": FakeDistance(),
        "config": FrozenTestConfig(expected_parent_count=2, flush_every=2),
        "deletion_fn": _deletions,
    }
    with pytest.raises(FrozenTestInterrupted):
        build_frozen_test_run(**kwargs, _interrupt_after_pairs=7)
    partial = [
        json.loads(line)
        for line in (output / "pair_matrix.jsonl").read_text().splitlines()
    ]
    assert len(partial) == 7

    summary = build_frozen_test_run(**kwargs)
    assert summary["actual_pair_rows"] == 40
    complete = [
        json.loads(line)
        for line in (output / "pair_matrix.jsonl").read_text().splitlines()
    ]
    assert len(complete) == 40
    assert len({(row["parent_id"], row["candidate_id"]) for row in complete}) == 40


def test_nonflip_or_null_distance_never_contributes_to_prefix():
    package_rows = [
        {"candidate_id": f"C{index:02d}", "canonical_fragment": fragment}
        for index, fragment in enumerate(FRAGMENTS, start=1)
    ]
    parents = [
        CalibrationParent("P1", "CCCO", 1, "test"),
        CalibrationParent("P2", "CCCN", 1, "test"),
    ]
    pair_rows = []
    for parent in parents:
        for candidate in package_rows:
            pair_rows.append(
                {
                    "parent_id": parent.parent_id,
                    "candidate_id": candidate["candidate_id"],
                    "applicable": True,
                    "pair_strict_flip": False,
                    "wnode_distance": None,
                    "cf_drop": None,
                }
            )
    # Load the threshold schema without fitting anything on these test rows.
    from src.eval.mutagenicity_wnode_frozen_test import _load_threshold_bundle

    thresholds = _load_threshold_bundle(_thresholds())
    metrics, _ = compute_frozen_prefix_metrics(
        pair_rows,
        parents,
        package_rows,
        thresholds,
    )
    assert all(row["strict_flip_parent_count"] == 0 for row in metrics)
    assert all(row["ccrcov_theta_star"] == 0.0 for row in metrics)
    assert all(row["fixed_capped_mean_cost"] == pytest.approx(0.07) for row in metrics)


def test_module_does_not_fit_test_quantiles():
    source = Path(
        "src/eval/mutagenicity_wnode_frozen_test.py"
    ).read_text(encoding="utf-8")
    assert "np.quantile(" not in source
    assert "derive_thresholds(" not in source
