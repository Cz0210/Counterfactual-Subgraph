import unittest
from src.baselines.comrecgc.rf_aligned_pool import predecessor_index, recorded_path, digest
from src.baselines.comrecgc.rf_aligned_recourse import storage_plan


def event(source, target, parent="P", action=None):
    return {"event": "selected_transition", "source_official_hash": source,
            "target_official_hash": target, "parent_id": parent,
            "action": action or ["EA", 0, 1], "action_resolution": "exact"}


class TestAidsPool(unittest.TestCase):
    def test_first_recorded_predecessor_not_later_graph_metadata(self):
        first = event("root", "a")
        index, count = predecessor_index([first, event("other", "a", "OTHER")])
        self.assertEqual(index["a"], first)
        self.assertEqual(count, 2)

    def test_order_and_full_action_count(self):
        index, _ = predecessor_index([event("root", "a"), event("a", "b")])
        row = {"official_graph_hash": "b", "parent_id": "P", "action_count": 2,
               "action_lineage_resolved": True}
        self.assertEqual([x["source_official_hash"] for x in recorded_path(row, index)], ["root", "a"])
        row["action_count"] = 1
        with self.assertRaisesRegex(ValueError, "count"):
            recorded_path(row, index)

    def test_no_action_inference(self):
        invalid = event("a", "b")
        invalid.pop("action")
        with self.assertRaisesRegex(ValueError, "inference is prohibited"):
            predecessor_index([invalid])

    def test_cross_parent_and_cycles_rejected(self):
        row = {"official_graph_hash": "b", "parent_id": "P", "action_count": 2,
               "action_lineage_resolved": True}
        with self.assertRaisesRegex(ValueError, "parent"):
            recorded_path(row, {"b": event("a", "b", "X")})
        with self.assertRaisesRegex(ValueError, "cycle"):
            recorded_path(row, {"b": event("b", "b")})

    def test_contract_hash_is_stable_and_scope_sensitive(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"denominator": 1283}), digest({"denominator": 1097}))

    def test_storage_bound_does_not_truncate_pair_universe(self):
        plan = storage_plan(parent_count=1097, candidate_count=100000, vector_dim=64, free_bytes=28 * 1024**3)
        self.assertEqual(plan["pair_rows_upper_bound"], 109700000)
        self.assertEqual(plan["state"], "BLOCKED_STORAGE")
        self.assertFalse(plan["pair_universe_truncated"])
        self.assertEqual(storage_plan(parent_count=1097, candidate_count=10, vector_dim=64, free_bytes=28 * 1024**3)["state"], "PASS")


class TestNativeReplayTensor(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            self.skipTest("Tiny tensor regression runs on the HPC CPU job")
        self.torch = torch
        self.graph = Data(x=torch.eye(3), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), num_nodes=3)
        self.graph.comrecgc_parent_id = "P"
        self.graph.comrecgc_source_smiles = "CCC"
        self.graph.smiles = "CCC"
        self.graph.comrecgc_node_origin = torch.tensor([0, 1, 2])

    def test_nr_target_ids_and_compact_roundtrip(self):
        from src.baselines.comrecgc.chem_repair import apply_action_to_graph
        from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
        from src.baselines.comrecgc.rf_aligned_pool import replay_candidate, compact_graph
        from src.baselines.comrecgc.rf_aligned_recourse import restore_graph
        target = apply_action_to_graph(self.graph, ["NR", 2])
        source_sha, target_sha = stable_untyped_graph_sha256(self.graph), stable_untyped_graph_sha256(target)
        transition = dict(event("root", "target", action=["NR", 2]), source_graph_sha256=source_sha, target_graph_sha256=target_sha, move_index=0, head_index=0)
        row = dict(candidate_index=0, official_graph_hash="target", parent_id="P", action_count=1, action_lineage_resolved=True, stable_graph_sha256=target_sha)
        replayed = replay_candidate(row, {"target": transition}, {"P": self.graph})
        self.assertEqual(replayed.comrecgc_node_origin.tolist(), [0, 1])
        self.assertEqual(self.graph.comrecgc_node_origin.tolist(), [0, 1, 2])
        record = dict(row, graph=compact_graph(replayed))
        restored = restore_graph(record, {"P": self.graph})
        self.assertTrue(self.torch.equal(restored.x, replayed.x))
        self.assertTrue(self.torch.equal(restored.edge_index, replayed.edge_index))
        row["stable_graph_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "candidate differs"):
            replay_candidate(row, {"target": transition}, {"P": self.graph})


if __name__ == "__main__":
    unittest.main()
