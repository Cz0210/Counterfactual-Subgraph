import unittest
from src.baselines.comrecgc.rf_aligned_pool import predecessor_index, recorded_path, digest


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


if __name__ == "__main__":
    unittest.main()
