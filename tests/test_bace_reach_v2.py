import copy
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval.bace_reach_selector import ReachMasks, select_nested


def thresholds():
    return {"theta_star": .1, "cost_cap": .5,
            "merged_thresholds": [{"threshold": .1, "weight": 3.}, {"threshold": .5, "weight": 1.}]}


class SelectorTests(unittest.TestCase):
    def fixture(self):
        d = np.full((30, 25), math.inf)
        for i in range(25):
            d[i, i] = .05 if i < 10 else .4
        d[25:, 20:] = .9
        return ReachMasks.from_distances([f"R{i:02}" for i in range(25)], d, thresholds())

    def test_nested_low_floor_and_fixed_prefixes(self):
        result = select_nested(self.fixture(), [f"R{i:02}" for i in range(20)], solver_seconds=0)
        self.assertGreaterEqual(result["new_S10_low_count"], result["old_S10_low_count"])
        order = result["ordered_rule_ids"]
        self.assertEqual(len(order), 20)
        self.assertEqual(len(set(order)), 20)
        for k in range(1, 21):
            self.assertEqual(result["prefixes"][str(k)], order[:k])
        self.assertFalse(result["test_loaded"])

    def test_deterministic_no_rng_consumption(self):
        rng = random.getstate()
        first = select_nested(self.fixture(), [f"R{i:02}" for i in range(20)], solver_seconds=0)
        self.assertEqual(rng, random.getstate())
        self.assertEqual(first, select_nested(self.fixture(), [f"R{i:02}" for i in range(20)], solver_seconds=0))

    def test_reach_precedes_high_cost_and_finite_reach(self):
        m = ReachMasks.from_distances(["cheap", "reach"], [[.01, 2.], [math.inf, 2.]], thresholds())
        self.assertGreater(m.score([1]), m.score([0]))
        self.assertEqual(m.score([1]), (2, 0, 0.))

    def test_nan_negative_duplicate_rejected(self):
        for ids, data in ((["x"], [[math.nan]]), (["x"], [[-.1]]), (["x", "x"], [[1, 2]])):
            with self.assertRaises(ValueError):
                ReachMasks.from_distances(ids, data, thresholds())

    def test_solver_feasible_small_case(self):
        result = select_nested(self.fixture(), [f"R{i:02}" for i in range(20)], solver_seconds=2)
        self.assertGreaterEqual(result["new_S10_low_count"], 10)
        self.assertFalse(result["global_nested_sequence_optimality_claimed"])


try:
    from rdkit import Chem
    from src.chem.bace_reach_search import (SearchBudget, all_matches, pattern_from_match,
        validate_attributed_candidate, search_parent, retain_train_pool, deletion_outcomes)
except ImportError:
    Chem = None


@unittest.skipUnless(Chem is not None, "RDKit is tested on the existing AutoDL environment")
class SearchTests(unittest.TestCase):
    def predict(self, smiles):
        return [{"predicted_label": 0 if Chem.MolFromSmiles(s).GetNumAtoms() <= 2 else 1,
                 "probabilities": [.8, .2] if Chem.MolFromSmiles(s).GetNumAtoms() <= 2 else [.2, .8]} for s in smiles]

    def test_attributed_aromatic_unsanitized_retained_and_tamper_rejected(self):
        m = Chem.MolFromSmiles("c1ccccc1CC")
        pattern = pattern_from_match(m, [0, 1])
        self.assertIsNotNone(validate_attributed_candidate(pattern))
        self.assertTrue(all_matches(m, pattern))
        bad = copy.deepcopy(pattern)
        bad["atoms"][0]["formal_charge"] += 1
        with self.assertRaisesRegex(ValueError, "BINDING_CONFLICT"):
            validate_attributed_candidate(bad)

    def test_beam_advances_beyond_seed_layer_and_query_budget(self):
        calls = []
        def predict(ss):
            calls.extend(ss)
            return self.predict(ss)
        state = search_parent(parent_id="train-1", parent_smiles="CCCCCCCC",
            before={"predicted_label": 1, "probabilities": [.1, .9]}, predict=predict,
            old_candidates=[], oracle_binding="frozen", budget=SearchBudget(), maximum_new_queries=128)
        self.assertTrue(state["witnesses"])
        self.assertTrue(any(len(r["match_atom_indices"]) > 2 for r in state["records"]))
        self.assertEqual(len(calls), state["new_graph_oracle_queries"])
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 128)
        self.assertFalse(state["test_loaded"])
        self.assertFalse(state["impossibility_proven"])

    def test_state_resume_and_rng_unchanged(self):
        rng = random.getstate()
        kwargs = dict(parent_id="train-1", parent_smiles="CCCCCCCC", before={"predicted_label": 1, "probabilities": [.1, .9]},
            predict=self.predict, old_candidates=[], oracle_binding="frozen", budget=SearchBudget())
        first = search_parent(**kwargs, maximum_new_queries=128)
        again = search_parent(**kwargs, maximum_new_queries=384, previous=first)
        self.assertEqual(first["new_graph_oracle_queries"], again["new_graph_oracle_queries"])
        self.assertEqual(rng, random.getstate())
        with self.assertRaisesRegex(ValueError, "BINDING_CHANGED"):
            search_parent(**{**kwargs, "oracle_binding": "changed"}, maximum_new_queries=384, previous=first)

    def test_old_pool_retained_no_cross_variant_injection(self):
        old = [{"candidate_id": "own", "canonical_fragment": "C"}]
        self.assertEqual(retain_train_pool(old, [], 20), old)

    def test_new_checker_all_matches_old_semantics(self):
        old = {"candidate_id": "own", "canonical_fragment": "CC"}
        new = pattern_from_match(Chem.MolFromSmiles("CCCC"), [0, 1])
        a = deletion_outcomes("CCCC", old, "p")
        b = deletion_outcomes("CCCC", new, "p")
        self.assertEqual([(r.match_atom_indices, r.residual_smiles, r.valid) for r in a],
                         [(r.match_atom_indices, r.residual_smiles, r.valid) for r in b])


if __name__ == "__main__":
    unittest.main()
