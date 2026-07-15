from __future__ import annotations

import math
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import globalgce_frequency_fgw_audit as audit


def _candidate(rank: int, smiles: str) -> audit.SelectedCandidate:
    return audit.SelectedCandidate(
        rank=rank,
        candidate_id=f"c{rank}",
        candidate_smiles=smiles,
        canonical_smiles=smiles,
        row_index=rank - 1,
        metadata={"frequency": 21 - rank, "graph_support": rank},
    )


def _row(
    parent: str,
    candidate: audit.SelectedCandidate,
    *,
    pred_before: int,
    pred_after: int,
    distance: float,
    recorded_flip: bool,
    method: str = "globalgce_frequency_top20",
    parent_smiles: str | None = None,
) -> dict[str, str]:
    return {
        "method": method,
        "parent_id": parent,
        "parent_smiles": parent_smiles or "",
        "candidate_id": candidate.candidate_id,
        "candidate_smiles": candidate.candidate_smiles,
        "label": "1",
        "pred_before": str(pred_before),
        "pred_after": str(pred_after),
        "cf_flip": str(recorded_flip),
        "teacher_strict_flip": str(recorded_flip),
        "cf_drop": "0.4",
        "distance": str(distance),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_audit_fixture(
    root: Path,
    *,
    parents: list[tuple[str, str]],
) -> dict[str, Path]:
    run_dir = root / "eval"
    selector_dir = root / "selector"
    report_dir = root / "report"
    output_dir = root / "audit"
    (run_dir / "details").mkdir(parents=True)
    (run_dir / "combined").mkdir(parents=True)
    selector_dir.mkdir()
    report_dir.mkdir()
    selected_rows = [
        {
            "rank": rank,
            "candidate_id": f"c{rank}",
            "candidate_smiles": "C" * rank,
            "frequency": 21 - rank,
            "graph_support": rank,
        }
        for rank in range(1, 21)
    ]
    selected_path = selector_dir / "selected_top20_for_eval.csv"
    _write_csv(selected_path, selected_rows)
    _write_csv(selector_dir / "selected_top20.csv", selected_rows)
    _write_csv(selector_dir / "frequency_ranked_candidates.csv", selected_rows)
    valid_rows: list[dict[str, object]] = []
    for row in selected_rows:
        for _occurrence in range(int(row["frequency"])):
            valid_rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "canonical_smiles": row["candidate_smiles"],
                    "raw_index": len(valid_rows),
                    "source_path": str(root / "globalgce_cfs_graphs.jsonl"),
                }
            )
    valid_path = root / "valid_candidates.csv"
    _write_csv(valid_path, valid_rows)
    pair_rows = [
        _row(
            parent_id,
            _candidate(rank, "C" * rank),
            pred_before=1,
            pred_after=0,
            distance=rank / 1000.0,
            recorded_flip=True,
            parent_smiles=parent_smiles,
        )
        for parent_id, parent_smiles in parents
        for rank in range(1, 21)
    ]
    _write_csv(run_dir / "details" / "pair_details.csv", pair_rows)
    _write_csv(
        run_dir / "combined" / "combined_threshold_summary.csv",
        [
            {
                "method": "globalgce_frequency_top20",
                "threshold": 0.0328,
                "close_cf_coverage": 1.0,
            }
        ],
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "fullgraph_method": "globalgce_frequency_top20",
                "fullgraph_candidates_path": str(selected_path),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cache_stats.json").write_text("{}", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "selected_path": selected_path,
        "valid_path": valid_path,
        "report_dir": report_dir,
        "output_dir": output_dir,
    }


def _fixture_args(paths: dict[str, Path], *extra: str) -> object:
    return audit.build_parser().parse_args(
        [
            "--run-dir",
            str(paths["run_dir"]),
            "--selected-top20",
            str(paths["selected_path"]),
            "--valid-candidates",
            str(paths["valid_path"]),
            "--report-dir",
            str(paths["report_dir"]),
            "--output-dir",
            str(paths["output_dir"]),
            "--method-name",
            "globalgce_frequency_top20",
            "--theta",
            "0.0328",
            *extra,
        ]
    )


def _write_ours_reference_run(
    run_dir: Path,
    parents: list[tuple[str, str]],
) -> None:
    rows = [
        {
            "method": "ours_selected_subgraphs",
            "parent_id": parent_id,
            "parent_smiles": parent_smiles,
        }
        for parent_id, parent_smiles in parents
    ]
    _write_csv(run_dir / "details" / "pair_details.csv", rows)


class GlobalGCEFrequencyFGWAuditTests(unittest.TestCase):
    def test_confusion_payload_exposes_canonical_totals(self) -> None:
        normalized = audit.normalize_strict_flip_confusion_payload(
            {
                "recorded_true_expected_true": 21940,
                "recorded_true_expected_false": 3720,
                "recorded_false_expected_true": 0,
                "recorded_false_expected_false": 0,
                "total_pair_rows": 25660,
                "recorded_true_pairs": 25660,
                "expected_strict_pairs": 21940,
                "mismatch_rows": 3720,
            }
        )
        self.assertEqual(normalized["mismatch_rows"], 3720)
        self.assertEqual(normalized["recorded_true_pairs"], 25660)
        self.assertEqual(normalized["expected_strict_pairs"], 21940)
        self.assertEqual(normalized["consistency_status"], "PASS")

    def test_historical_confusion_payload_infers_missing_mismatch(self) -> None:
        normalized = audit.normalize_strict_flip_confusion_payload(
            {
                "confusion": {
                    "recorded_true_expected_true": 21940,
                    "recorded_true_expected_false": 3720,
                    "recorded_false_expected_true": 0,
                    "recorded_false_expected_false": 0,
                },
                "rows_after_reference_filter": 25660,
                "recorded_cf_flip_pairs": 25660,
                "expected_strict_flip_pairs": 21940,
            }
        )
        self.assertEqual(normalized["mismatch_rows"], 3720)
        self.assertEqual(normalized["consistency_status"], "PASS_WITH_WARNINGS")
        self.assertIn(
            "inferred_missing_field:mismatch_rows",
            normalized["consistency_warnings"],
        )

    def test_weak_flip_saved_row_is_reported_as_mismatch(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row(
                "p0",
                selected[0],
                pred_before=0,
                pred_after=0,
                distance=0.01,
                recorded_flip=True,
            )
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        self.assertEqual(result.recorded_flip_pairs, 1)
        self.assertEqual(result.expected_flip_pairs, 0)
        self.assertEqual(len(result.mismatch_rows), 1)
        self.assertTrue(result.mismatch_rows[0]["old_weak_flip"])
        self.assertEqual(result.confusion["recorded_true_expected_false"], 1)
        self.assertEqual(result.recorded_flip_pairs, sum(
            result.confusion[key]
            for key in ("recorded_true_expected_true", "recorded_true_expected_false")
        ))

    def test_confusion_matrix_counts_false_positive_and_false_negative(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row("tp", selected[0], pred_before=1, pred_after=0, distance=0.1, recorded_flip=True),
            _row("fp", selected[0], pred_before=0, pred_after=0, distance=0.1, recorded_flip=True),
            _row("fn", selected[0], pred_before=1, pred_after=0, distance=0.1, recorded_flip=False),
            _row("tn", selected[0], pred_before=1, pred_after=1, distance=0.1, recorded_flip=False),
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        self.assertEqual(
            result.confusion,
            {
                "recorded_true_expected_true": 1,
                "recorded_true_expected_false": 1,
                "recorded_false_expected_true": 1,
                "recorded_false_expected_false": 1,
            },
        )
        self.assertEqual(len(result.mismatch_rows), 2)

    def test_recorded_all_true_mismatch_equals_total_minus_expected(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row(f"p{index}", selected[0], pred_before=before, pred_after=0, distance=0.1, recorded_flip=True)
            for index, before in enumerate((1, 1, 0, 0))
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        self.assertEqual(result.recorded_flip_pairs, 4)
        self.assertEqual(result.expected_flip_pairs, 2)
        self.assertEqual(len(result.mismatch_rows), 2)
        self.assertEqual(len(result.mismatch_rows), len(rows) - result.expected_flip_pairs)

    def test_corrected_pair_rows_replace_weak_flip_without_recomputing_distance(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row("p0", selected[0], pred_before=0, pred_after=0, distance=0.0123, recorded_flip=True)
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        corrected = audit.corrected_pair_rows(result, target_label=1)[0]
        self.assertFalse(corrected["teacher_strict_flip"])
        self.assertFalse(corrected["cf_flip"])
        self.assertTrue(corrected["old_weak_flip"])
        self.assertEqual(corrected["distance"], "0.0123")

    def test_method_and_reference_parent_filters_define_one_consistent_cohort(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row("keep1", selected[0], pred_before=1, pred_after=0, distance=0.01, recorded_flip=True),
            _row("keep2", selected[0], pred_before=0, pred_after=0, distance=0.01, recorded_flip=True),
            _row("extra", selected[0], pred_before=0, pred_after=0, distance=0.01, recorded_flip=True),
            _row("other", selected[0], pred_before=0, pred_after=0, distance=0.01, recorded_flip=True, method="ours_selected_subgraphs"),
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
            reference_parent_ids={"keep1", "keep2"},
        )
        self.assertEqual(result.rows_before_method_filter, 4)
        self.assertEqual(result.rows_after_method_filter, 3)
        self.assertEqual(result.rows_after_reference_filter, 2)
        self.assertEqual(result.parents, {"keep1", "keep2"})
        self.assertEqual(len(result.mismatch_rows), 1)
        self.assertEqual({row["method"] for row in result.mismatch_rows}, {"globalgce_frequency_top20"})

    def test_reference_filter_keeps_exactly_1283_parent_ids(self) -> None:
        selected = [_candidate(1, "C")]
        reference = {str(index) for index in range(1283)}
        rows = [
            _row(parent, selected[0], pred_before=1, pred_after=0, distance=0.01, recorded_flip=True)
            for parent in [*sorted(reference, key=int), "extra"]
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
            reference_parent_ids=reference,
        )
        self.assertEqual(len(result.parents), 1283)
        self.assertNotIn("extra", result.parents)

    def test_strict_flip_and_prefix_marginal_coverage(self) -> None:
        selected = [_candidate(1, "C"), _candidate(2, "N"), _candidate(3, "O")]
        rows = [
            _row("p1", selected[0], pred_before=1, pred_after=0, distance=0.01, recorded_flip=True),
            _row("p2", selected[0], pred_before=1, pred_after=0, distance=0.08, recorded_flip=True),
            _row("p1", selected[1], pred_before=1, pred_after=0, distance=0.02, recorded_flip=True),
            _row("p2", selected[1], pred_before=1, pred_after=0, distance=0.03, recorded_flip=True),
            _row("p3", selected[2], pred_before=0, pred_after=0, distance=0.01, recorded_flip=False),
        ]
        result = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        marginal = audit.build_prefix_marginal_rows(result, selected, theta=0.05)
        self.assertEqual([row["marginal_newly_covered_parent_count"] for row in marginal], [1, 1, 0])
        self.assertEqual([row["cumulative_covered_parent_count"] for row in marginal], [1, 2, 2])
        self.assertLessEqual(marginal[1]["median_fgw_distance_on_standalone_covered"], 0.05)

    def test_candidate_order_audit_detects_reordering(self) -> None:
        selected = [_candidate(1, "C"), _candidate(2, "N")]
        rows = [
            _row("p1", selected[1], pred_before=1, pred_after=0, distance=0.01, recorded_flip=True),
            _row("p1", selected[0], pred_before=1, pred_after=0, distance=0.02, recorded_flip=True),
        ]
        pair_audit = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        order_rows, summary = audit.audit_candidate_order(selected, pair_audit)
        self.assertFalse(summary["rank_order_preserved"])
        self.assertEqual(order_rows[0]["evaluator_internal_index"], 1)

    def test_frequency_uses_all_raw_rows_grouped_by_canonical_smiles(self) -> None:
        rows = [
            {"canonical_smiles": "C", "raw_index": "1", "source_path": "a.jsonl", "selected": "True", "duplicate": "False"},
            {"canonical_smiles": "C", "raw_index": "2", "source_path": "a.jsonl", "selected": "False", "duplicate": "True"},
            {"canonical_smiles": "N", "raw_index": "3", "source_path": "a.jsonl", "selected": "False", "duplicate": "False"},
        ]
        groups, summary = audit.build_frequency_groups(rows)
        self.assertEqual(groups["C"]["raw_frequency"], 2)
        self.assertEqual(groups["C"]["source_raw_index_count"], 2)
        self.assertEqual(summary["raw_rows"], 3)
        self.assertFalse(summary["frequency_counts_selected_only"])
        self.assertFalse(summary["frequency_counts_duplicate_false_only"])

    def test_covered_median_is_theta_conditioned_but_applicable_median_is_not(self) -> None:
        selected = [_candidate(1, "C")]
        rows = [
            _row("p1", selected[0], pred_before=1, pred_after=0, distance=0.02, recorded_flip=True),
            _row("p2", selected[0], pred_before=1, pred_after=0, distance=0.08, recorded_flip=True),
            _row("p3", selected[0], pred_before=0, pred_after=0, distance=0.01, recorded_flip=False),
            _row("p4", selected[0], pred_before=0, pred_after=0, distance=0.02, recorded_flip=False),
        ]
        pair_audit = audit.audit_pairs(
            rows,
            method="globalgce_frequency_top20",
            selected=selected,
            target_label=1,
        )
        metrics = audit._prefix_metrics(pair_audit, selected, k=1, theta=0.0328)
        self.assertEqual(metrics["num_covered"], 1)
        self.assertAlmostEqual(metrics["covered_parent_median_cost"], 0.02)
        self.assertAlmostEqual(metrics["applicable_parent_median_cost"], 0.05)
        self.assertTrue(math.isinf(metrics["unconditional_median_cost"]))

    def test_default_production_ours_path_is_not_read_without_explicit_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _build_audit_fixture(
                root,
                parents=[("p1", "CCO"), ("p2", "CCN")],
            )
            fake_production = root / "production_ours"
            _write_ours_reference_run(fake_production, [("0", "CCC")])
            with mock.patch.object(
                audit,
                "DEFAULT_COMPARISON_RUNS",
                {"Ours": str(fake_production)},
            ):
                summary = audit.run_audit(_fixture_args(paths))
            self.assertEqual(
                summary["reference_cohort"]["source_kind"],
                "all_label_parent_diagnostic",
            )
            self.assertIsNone(summary["reference_cohort"]["source_path"])
            self.assertEqual(summary["strict_flip_audit"]["num_parents"], 2)

    def test_explicit_ours_comparison_enables_reference_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _build_audit_fixture(
                root,
                parents=[("g1", "CCO"), ("g2", "CCN")],
            )
            ours_run = root / "ours"
            _write_ours_reference_run(ours_run, [("g1", "CCO"), ("g2", "CCN")])
            summary = audit.run_audit(
                _fixture_args(
                    paths,
                    "--comparison-run",
                    f"Ours={ours_run}",
                    "--expected-reference-parents",
                    "2",
                )
            )
            self.assertEqual(
                summary["reference_cohort"]["source_kind"],
                "explicit_auto_reference_from_ours_with_crosswalk",
            )
            self.assertEqual(summary["reference_cohort"]["num_reference_parents"], 2)

    def test_explicit_reference_parent_csv_uses_target_namespace_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _build_audit_fixture(
                root,
                parents=[("g1", "CCO"), ("g2", "CCN")],
            )
            reference_path = root / "reference_parent_ids_globalgce_namespace.csv"
            _write_csv(
                reference_path,
                [
                    {
                        "parent_id": "g1",
                        "source_ours_parent_id": "0",
                        "parent_smiles": "CCO",
                        "canonical_smiles": "CCO",
                        "match_type": "canonical_smiles",
                    },
                    {
                        "parent_id": "g2",
                        "source_ours_parent_id": "1",
                        "parent_smiles": "CCN",
                        "canonical_smiles": "CCN",
                        "match_type": "canonical_smiles",
                    },
                ],
            )
            summary = audit.run_audit(
                _fixture_args(
                    paths,
                    "--reference-parent-ids",
                    str(reference_path),
                    "--reference-parent-id-col",
                    "parent_id",
                    "--expected-reference-parents",
                    "2",
                )
            )
            self.assertEqual(
                summary["reference_cohort"]["source_kind"],
                "explicit_reference_parent_ids",
            )
            self.assertNotIn("crosswalk_applied", summary["reference_cohort"])
            written = (paths["output_dir"] / "reference_parent_ids.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("g1", written)
            self.assertIn("g2", written)

    def test_explicit_auto_reference_crosswalks_different_id_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _build_audit_fixture(
                root,
                parents=[("raw10", "CCO"), ("raw20", "CCN")],
            )
            ours_run = root / "ours"
            _write_ours_reference_run(ours_run, [("0", "CCO"), ("1", "CCN")])
            summary = audit.run_audit(
                _fixture_args(
                    paths,
                    "--auto-reference-from-ours",
                    "--reference-ours-run",
                    str(ours_run),
                    "--expected-reference-parents",
                    "2",
                )
            )
            self.assertTrue(summary["reference_cohort"]["crosswalk_applied"])
            self.assertEqual(summary["strict_flip_audit"]["num_parents"], 2)
            crosswalk_rows = audit._read_csv(
                paths["output_dir"] / "reference_parent_crosswalk.csv"
            )[1]
            self.assertEqual(
                {row["parent_id"] for row in crosswalk_rows},
                {"raw10", "raw20"},
            )
            self.assertEqual(
                {row["source_ours_parent_id"] for row in crosswalk_rows},
                {"0", "1"},
            )

    def test_no_reference_summary_source_paths_stay_inside_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _build_audit_fixture(
                root,
                parents=[("p1", "CCO"), ("p2", "CCN")],
            )
            summary = audit.run_audit(_fixture_args(paths))
            source_paths: list[str] = []

            def collect(value: object) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "source_path" and isinstance(item, str) and item:
                            source_paths.append(item)
                        collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)

            collect(summary)
            self.assertTrue(all(Path(path).is_relative_to(root) for path in source_paths))

    def test_end_to_end_audit_writes_required_outputs_without_distance_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "eval"
            selector_dir = root / "selector"
            report_dir = root / "report"
            output_dir = root / "audit"
            (run_dir / "details").mkdir(parents=True)
            (run_dir / "combined").mkdir(parents=True)
            selector_dir.mkdir()
            report_dir.mkdir()

            selected_rows = [
                {
                    "rank": rank,
                    "candidate_id": f"c{rank}",
                    "candidate_smiles": "C" * rank,
                    "frequency": 21 - rank,
                    "graph_support": rank,
                }
                for rank in range(1, 21)
            ]

            def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            selected_path = selector_dir / "selected_top20_for_eval.csv"
            write_csv(selected_path, selected_rows)
            write_csv(selector_dir / "selected_top20.csv", selected_rows)
            write_csv(selector_dir / "frequency_ranked_candidates.csv", selected_rows)
            valid_rows: list[dict[str, object]] = []
            for row in selected_rows:
                for _occurrence in range(int(row["frequency"])):
                    valid_rows.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "canonical_smiles": row["candidate_smiles"],
                            "raw_index": len(valid_rows),
                            "source_path": "globalgce_cfs_graphs.jsonl",
                        }
                    )
            valid_path = root / "valid_candidates.csv"
            write_csv(valid_path, valid_rows)

            pair_rows = [
                _row(
                    parent,
                    _candidate(rank, "C" * rank),
                    pred_before=1,
                    pred_after=0,
                    distance=rank / 1000.0,
                    recorded_flip=True,
                )
                for parent in ("p1", "p2")
                for rank in range(1, 21)
            ]
            write_csv(run_dir / "details" / "pair_details.csv", pair_rows)
            write_csv(
                run_dir / "combined" / "combined_threshold_summary.csv",
                [
                    {
                        "method": "globalgce_frequency_top20",
                        "threshold": 0.0328,
                        "close_cf_coverage": 1.0,
                    }
                ],
            )
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "fullgraph_method": "globalgce_frequency_top20",
                        "fullgraph_candidates_path": str(selected_path),
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "cache_stats.json").write_text("{}", encoding="utf-8")
            args = audit.build_parser().parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--selected-top20",
                    str(selected_path),
                    "--valid-candidates",
                    str(valid_path),
                    "--report-dir",
                    str(report_dir),
                    "--output-dir",
                    str(output_dir),
                    "--method-name",
                    "globalgce_frequency_top20",
                    "--theta",
                    "0.0328",
                ]
            )
            summary = audit.run_audit(args)
            self.assertFalse(summary["distance_recomputed"])
            self.assertEqual(summary["strict_flip_audit"]["num_parents"], 2)
            self.assertEqual(summary["strict_flip_audit"]["num_pair_rows"], 40)
            self.assertEqual(
                summary["reference_cohort"]["source_kind"],
                "all_label_parent_diagnostic",
            )
            self.assertIn(
                "reference_parent_cohort_unavailable_running_all_label_parent_diagnostic",
                summary["warnings"],
            )
            self.assertTrue(summary["candidate_order_audit"]["rank_order_preserved"])
            confusion = json.loads(
                (output_dir / "strict_flip_confusion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(confusion["consistency_status"], "PASS")
            self.assertEqual(
                confusion["confusion"]["mismatch_rows"],
                confusion["mismatch_rows"],
            )
            for field in (
                "recorded_true_expected_true",
                "recorded_true_expected_false",
                "recorded_false_expected_true",
                "recorded_false_expected_false",
                "recorded_true_pairs",
                "expected_strict_pairs",
                "mismatch_rows",
            ):
                self.assertIn(field, confusion)
            for name in (
                "audit_summary.json",
                "audit_report.txt",
                "globalgce_prefix_marginal_coverage.csv",
                "candidate_order_audit.csv",
                "strict_flip_mismatches.csv",
                "strict_flip_confusion.json",
                "parent_cohort_audit.csv",
                "reference_parent_ids.csv",
                "reference_parent_crosswalk.csv",
                "corrected_table2_metrics.csv",
                "metric_definition_audit.json",
                "applicable_rate_audit.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
