from __future__ import annotations

import math
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


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
) -> dict[str, str]:
    return {
        "method": method,
        "parent_id": parent,
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


class GlobalGCEFrequencyFGWAuditTests(unittest.TestCase):
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
            self.assertTrue(summary["candidate_order_audit"]["rank_order_preserved"])
            for name in (
                "audit_summary.json",
                "audit_report.txt",
                "globalgce_prefix_marginal_coverage.csv",
                "candidate_order_audit.csv",
                "strict_flip_mismatches.csv",
                "metric_definition_audit.json",
                "applicable_rate_audit.csv",
            ):
                self.assertTrue((output_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
