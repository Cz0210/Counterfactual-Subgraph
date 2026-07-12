from __future__ import annotations

import unittest

from scripts.convert_gcf_hiv_csv_graphs_to_smiles import _graph_entries
from scripts.gcf_hiv_csv_export_summary import _finite_float_or_default, _greedy
from scripts.gcf_hiv_csv_export_valid_greedy_topk import (
    build_aligned_outputs,
    filter_valid_candidates,
    select_valid_greedy_topk,
)


def _valid_smiles(smiles: str) -> tuple[bool, str, str, int]:
    return True, "ok", smiles, 2


class GCFHIVCSVValidGreedyTests(unittest.TestCase):
    def test_invalid_candidate_never_updates_covered_set(self) -> None:
        records = [
            {
                "candidate_id": "invalid",
                "graph_hash": "h-invalid",
                "frequency": 100,
                "min_distance_seen": 0.0,
                "covered_indices": {0, 1, 2},
                "graph": "graph-invalid",
            },
            {
                "candidate_id": "valid",
                "graph_hash": "h-valid",
                "frequency": 1,
                "min_distance_seen": 0.5,
                "covered_indices": {3},
                "graph": "graph-valid",
            },
        ]
        validity = [
            {
                "candidate_id": "invalid",
                "graph_hash": "h-invalid",
                "smiles": "bad",
                "convert_ok": "true",
                "sanitize_ok": "false",
            },
            {
                "candidate_id": "valid",
                "graph_hash": "h-valid",
                "smiles": "CC",
                "convert_ok": "true",
                "sanitize_ok": "true",
            },
        ]
        legal, reasons = filter_valid_candidates(
            records,
            validity,
            validate_smiles_fn=_valid_smiles,
        )
        selected = select_valid_greedy_topk(legal, top_k=1, seed=0)
        self.assertEqual([row["candidate_id"] for row in selected], ["valid"])
        self.assertEqual(selected[0]["covered_count_at_rank"], 1)
        self.assertEqual(selected[0]["covered_indices"], {3})
        self.assertEqual(reasons["sanitize_mapping_failed"], 1)

    def test_zero_min_distance_is_preserved_and_preferred(self) -> None:
        records = [
            {
                "candidate_id": "positive-distance",
                "frequency": 5,
                "min_distance_seen": 0.2,
                "covered_indices": {0},
            },
            {
                "candidate_id": "zero-distance",
                "frequency": 5,
                "min_distance_seen": 0.0,
                "covered_indices": {1},
            },
        ]
        selected = _greedy(records, 1)
        self.assertEqual(selected[0]["candidate_id"], "zero-distance")
        self.assertEqual(_finite_float_or_default(0.0, 999.0), 0.0)
        self.assertEqual(_finite_float_or_default(float("nan"), 999.0), 999.0)

    def test_graph_metadata_and_smiles_orders_are_identical(self) -> None:
        selected = [
            {
                "selected_rank": 1,
                "candidate_id": "c2",
                "graph_hash": "h2",
                "smiles": "CC",
                "frequency": 2,
                "min_distance_seen": 0.0,
                "marginal_coverage_gain": 2,
                "covered_count_at_rank": 2,
                "atom_count": 2,
                "graph": "graph-2",
            },
            {
                "selected_rank": 2,
                "candidate_id": "c1",
                "graph_hash": "h1",
                "smiles": "CO",
                "frequency": 1,
                "min_distance_seen": 0.1,
                "marginal_coverage_gain": 1,
                "covered_count_at_rank": 3,
                "atom_count": 2,
                "graph": "graph-1",
            },
        ]
        graphs, metadata, smiles = build_aligned_outputs(selected, num_parents=4)
        self.assertEqual(graphs, ["graph-2", "graph-1"])
        self.assertEqual([row["candidate_id"] for row in metadata], ["c2", "c1"])
        self.assertEqual([row["candidate_id"] for row in smiles], ["c2", "c1"])
        self.assertEqual([row["rank"] for row in metadata], [1, 2])

    def test_same_input_and_seed_are_deterministic(self) -> None:
        records = [
            {
                "candidate_id": f"c{index}",
                "frequency": 1,
                "min_distance_seen": 0.1,
                "covered_indices": {index},
            }
            for index in range(4)
        ]
        first = select_valid_greedy_topk(records, top_k=4, seed=13)
        second = select_valid_greedy_topk(records, top_k=4, seed=13)
        self.assertEqual(
            [row["candidate_id"] for row in first],
            [row["candidate_id"] for row in second],
        )

    def test_raw_counterfactual_payload_preserves_candidate_order(self) -> None:
        payload = {
            "counterfactual_candidates": [
                {"graph_hash": "h2"},
                {"graph_hash": "h1"},
            ],
            "graph_map": {"h1": "graph-1", "h2": "graph-2"},
        }
        entries = _graph_entries(payload)
        self.assertEqual([graph for graph, _ in entries], ["graph-2", "graph-1"])
        self.assertEqual(
            [record["candidate_id"] for _, record in entries],
            ["gcf_hiv_csv_0", "gcf_hiv_csv_1"],
        )


if __name__ == "__main__":
    unittest.main()
