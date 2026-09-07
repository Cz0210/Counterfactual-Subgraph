import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src.eval.bace_reach_closeout import retained_support_gate, reach_parent, freeze_final, verify_witness_application, run_cpu_closeout
    from src.eval.bace_reach_v2 import seal
    from src.chem.bace_reach_search import pattern_from_match
    from rdkit import Chem
except ImportError:
    Chem = None


@unittest.skipUnless(Chem is not None, "Existing AutoDL CPU environment supplies RDKit")
class CloseoutTests(unittest.TestCase):
    def row(self, pid, candidate="new"):
        return {"parent_id": pid, "before": {"predicted_label": 1}, "old_pool_pairs": [],
                "search": {"witnesses": [{"pattern": {"candidate_id": candidate}, "valid": True,
                                            "strict_flip": True, "after": {"predicted_label": 0}}]}}

    def test_retained_pool_gate_not_uncapped_witness_count(self):
        rows = [self.row(str(i)) for i in range(10)]
        rows[-1] = self.row("9", "discarded")
        result = retained_support_gate(rows, [{"candidate_id": "new"}])
        self.assertEqual(result["retained_verified_reach_lower_bound"], .9)
        self.assertEqual(result["state"], "NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE")
        self.assertFalse(result["full_pool_reach_exact"])
        missing = retained_support_gate(rows, [])
        self.assertEqual(missing["retained_verified_reach_lower_count"], 0)
        self.assertEqual(missing["state"], "NEEDS_RETAINED_POOL_CROSS_PARENT_REACH_NOT_PPO_AUTHORIZED")

    def test_source_denominator_and_duplicate_parent(self):
        rows = [self.row("eligible"), {**self.row("wrong-source"), "before": {"predicted_label": 0}}]
        self.assertEqual(retained_support_gate(rows, [{"candidate_id": "new"}])["source_eligible_count"], 1)
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            retained_support_gate([self.row("x"), self.row("x")], [])

    def test_full_pool_reach_only_oracle_no_ot_and_real_witness(self):
        parent = SimpleNamespace(parent_id="p", smiles="CCC")
        pattern = pattern_from_match(Chem.MolFromSmiles("CCC"), [0])
        calls = []
        def predict(smiles):
            calls.extend(smiles)
            return [{"predicted_label": 0, "probabilities": [.8, .2]} for _ in smiles]
        result = reach_parent(parent=parent, candidates=[pattern], before={"predicted_label": 1},
                              predict=predict, oracle_binding="oracle")
        self.assertTrue(result["reachable"])
        self.assertEqual(result["witness"]["residual_smiles"], "CC")
        self.assertEqual(len(calls), 1)
        known = [{"delete_valid": True, "residual_smiles": "CC", "p_after": [.8,.2],
                  "pred_after": 0, "oracle_checkpoint_hash": "oracle"}]
        cached = reach_parent(parent=parent, candidates=[pattern], before={"predicted_label": 1},
                              predict=lambda _: self.fail("cached graph re-inferred"), oracle_binding="oracle", known_rows=known)
        self.assertEqual(cached["new_graph_oracle_queries"], 0)

    def test_mask_witness_must_reapply_as_retained_rule(self):
        pattern = pattern_from_match(Chem.MolFromSmiles("CCC"), [0])
        witness = {"match_atom_indices": [0], "residual_smiles": "CC", "strict_flip": True,
                   "valid": True, "before": {"predicted_label": 1}, "after": {"predicted_label": 0}}
        self.assertTrue(verify_witness_application("CCC", witness, pattern))
        with self.assertRaisesRegex(ValueError, "NOT_REPLAYABLE"):
            verify_witness_application("CCC", {**witness, "residual_smiles": "C"}, pattern)

    def test_single_final_root_freeze_without_opening_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "reference.json"
            ref.write_text(json.dumps({"frozen_downstream": {"dataset_split_paths": {"test": str(root/"DO_NOT_OPEN_TEST")},
                                                               "dataset_split_hashes": {"test": "a"*64}}}))
            thresholds = root / "thresholds.json"
            thresholds.write_text("{}")
            contract = seal(root / "search_contract.json", {"paths": {"reference": str(ref), "thresholds": str(thresholds)}})
            pool = seal(root / "candidate_freeze.json", {"state": "TRAIN_ONLY_POOL_FROZEN"})
            order = [f"R{i}" for i in range(20)]
            seal(root / "selector_freeze.json", {"candidate_freeze_sha256": pool["self_sha256"], "test_opened": False,
                 "controls": {"old_pool_old_selector": order, "new_pool_old_selector": order},
                 "reach_first": {"ordered_rule_ids": order}})
            seal(root / "train_reach_gate.json", {"candidate_freeze_sha256": pool["self_sha256"],
                                                "state": "NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE"})
            descriptor = {"path": "unopened-old-pairs", "sha256": "b"*64,
                          "receipt_path": "unopened-old-receipt", "receipt_sha256": "c"*64}
            raw = {"portable_manifest": {"path": "unopened-small-metadata", "sha256": "d"*64},
                   "source_spec": {"path": "unopened-source-spec", "sha256": "e"*64}}
            frozen = freeze_final(root, root/"final", old_test_pair_source=descriptor, raw_distance_source=raw)
            self.assertFalse(frozen["test_opened"])
            self.assertEqual(frozen["selected_control"], "new_pool_reach_first")
            self.assertEqual(len(frozen["controls"]), 3)
            self.assertEqual(freeze_final(root, root/"final", old_test_pair_source=descriptor, raw_distance_source=raw), frozen)
            with self.assertRaisesRegex(ValueError, "IMMUTABLE_REACH_RECEIPT_EXISTS"):
                freeze_final(root, root/"second-final", old_test_pair_source=descriptor, raw_distance_source=raw)

    def test_cpu_successor_freezes_before_waiting_for_test_index(self):
        with tempfile.TemporaryDirectory() as temp:
            campaign = Path(temp)/"campaign"
            campaign.mkdir()
            (campaign/"selector_freeze.json").write_text("{}"); calls = []
            index = Path(temp)/"index-descriptor.json"
            def freeze(*args, **kwargs):
                calls.append("freeze")
                return seal(campaign/"final_test_binding.json", {"test_opened": False})
            def wait(seconds):
                self.assertTrue((campaign/"final_test_binding.json").is_file())
                calls.append("index_ready")
                index.write_text('{}')
            def evaluate(*args, **kwargs):
                calls.append("test")
                return {"self_sha256": "final"}
            with patch("src.eval.bace_reach_closeout.train_gate", return_value={"state": "NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE"}), \
                 patch("src.eval.bace_reach_closeout.freeze_final", side_effect=freeze), \
                 patch("src.eval.bace_reach_closeout.run_final_test", side_effect=evaluate), \
                 patch("src.eval.bace_reach_closeout.time.sleep", side_effect=wait):
                run_cpu_closeout(campaign, Path(temp)/"final", old_test_pair_source={},
                    raw_distance_source={}, raw_test_index_descriptor=index, boundary_check=lambda: None)
            self.assertEqual(calls, ["freeze", "index_ready", "test"])
            terminal = json.loads((Path(temp)/"final/cpu_owner_terminal.json").read_text())
            self.assertFalse(terminal["gpu_requested"])
            self.assertTrue(terminal["independent_audit_still_required"])


if __name__ == "__main__":
    unittest.main()
