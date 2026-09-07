import copy
from collections import Counter
from dataclasses import asdict
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.chem.bace_reach_search import SearchBudget, new_graph_key, search_parent, retain_train_pool
from src.eval.bace_frozen_gnn_contracts import stable_sha256
from src.experiments import bace_gin_reach_supplement as leaf


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.parent = SimpleNamespace(parent_id="train1", smiles="CCCC", label=1)
        self.adoption = dict(model_sha256="gin-weight", temperature_sha256="fit-temp",
            temperature=1.2, feature_schema_sha256="schema")
        self.before = dict(predicted_label=1, probabilities=[.1, .9])

    def unit(self):
        common = dict(parent_id="train1", parent_smiles="CCCC", candidate_id="rule",
            pred_before=1, p1_before=.9, split="train", oracle_checkpoint_hash="gin-weight",
            oracle_backbone="gin", oracle_temperature=1.2, rf_oracle_used=False)
        return dict(spec_sha256="spec", parent_id="train1", split="train", group="adopted2607",
            old_gine_flip_masks_reused=False, binding=stable_sha256(dict(spec="spec", group="adopted2607",
                split="train", parent_id="train1", pool=["rule"], freeze=None)),
            pair_rows=[dict(common, num_matches=1, num_strict_flip_matches=0)],
            match_rows=[dict(common, p_before=[.1, .9], delete_valid=True, residual_smiles="CCC",
                residual_connected=True, sanitize_ok=True, residual_num_components=1, contains_dot=False,
                p_after=[.2, .8], pred_after=1, teacher_strict_flip=False)])

    def inspect(self, unit):
        return leaf.inspect_parent(unit, spec_sha="spec", parent=self.parent,
            candidate_ids=["rule"], adoption=self.adoption)

    def test_exact_own_gin_residual_cache_and_source(self):
        info = self.inspect(self.unit())
        key = new_graph_key("CCC", leaf.oracle_identity(self.adoption))
        self.assertEqual(info["cache"][key], dict(predicted_label=1, probabilities=[.2, .8]))
        self.assertEqual(info["before"], self.before)
        self.assertFalse(info["covered"])

    def test_wrong_oracle_temperature_split_or_pair_scope_rejected(self):
        for key, value in (("oracle_checkpoint_hash", "gine"), ("oracle_temperature", 1.0),
                           ("oracle_backbone", "gine"), ("parent_smiles", "CC"), ("split", "test")):
            bad = self.unit()
            bad["match_rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.inspect(bad)
        bad = self.unit()
        bad["pair_rows"] = []
        with self.assertRaisesRegex(ValueError, "COVERAGE_INCOMPLETE"):
            self.inspect(bad)

    def test_match_count_gap_and_false_flip_rejected(self):
        for mutate in (lambda u: u["pair_rows"][0].update(num_matches=2),
                       lambda u: u["match_rows"][0].update(teacher_strict_flip=True)):
            bad = self.unit()
            mutate(bad)
            with self.assertRaises(ValueError):
                self.inspect(bad)

    def test_source_reach_is_strict_flip_not_distance_threshold(self):
        unit = self.unit()
        unit["match_rows"][0].update(p_after=[.8, .2], pred_after=0, teacher_strict_flip=True,
            wnode_distance=999.0)
        unit["pair_rows"][0]["num_strict_flip_matches"] = 1
        self.assertTrue(self.inspect(unit)["covered"])

    def test_conflicting_saved_graph_not_averaged_or_reused(self):
        unit = self.unit()
        unit["match_rows"].append(dict(unit["match_rows"][0], p_after=[.21, .79]))
        unit["pair_rows"][0]["num_matches"] = 2
        info = self.inspect(unit)
        self.assertEqual(info["cache"], {})
        self.assertEqual(info["conflicting_saved_graphs"], 1)

    def test_seed7_extra_order_local_rng_stable_at_most128(self):
        states = [dict(parent_id=f"p{i:03}", witnesses=[] if i < 140 else [1]) for i in range(150)]
        original = random.getstate()
        actual = leaf.seeded_extra_ids(states)
        expected = [f"p{i:03}" for i in range(140)]
        random.Random(7).shuffle(expected)
        self.assertEqual(actual, expected[:128])
        self.assertEqual(actual, leaf.seeded_extra_ids(list(reversed(states))))
        self.assertEqual(random.getstate(), original)
        self.assertNotEqual(actual, sorted(actual))

    def test_initial_saved_predictions_do_not_spend_new_queries_or_mutate(self):
        calls = []
        seed = {new_graph_key("CCC", "gin"): dict(predicted_label=1, probabilities=[.2, .8])}
        previous_seed = copy.deepcopy(seed)
        def predict(smiles):
            calls.extend(smiles)
            return [dict(predicted_label=0, probabilities=[.8, .2]) for _ in smiles]
        args = dict(parent_id="p", parent_smiles="CCCC", before=self.before, predict=predict,
            old_candidates=[], oracle_binding="gin", budget=SearchBudget(), source_label=1)
        first = search_parent(**args, maximum_new_queries=128, initial_oracle_cache=seed)
        self.assertNotIn("CCC", calls)
        self.assertGreater(first["pass_cache_hits"], 0)
        self.assertEqual(first["new_graph_oracle_queries"], len(calls))
        self.assertEqual(seed, previous_seed)
        with self.assertRaisesRegex(ValueError, "INITIAL_CACHE_ONLY"):
            search_parent(**args, maximum_new_queries=384, previous=first, initial_oracle_cache=seed)

    def test_original2607_preserved_without_generation_or_extra_raw_seeds(self):
        old = [dict(candidate_id=f"r{i}", canonical_fragment=f"old-{i}") for i in range(2607)]
        before = copy.deepcopy(old)
        self.assertEqual(retain_train_pool(old, []), before)
        states = [dict(parent_id="p", witnesses=[dict(pattern=dict(candidate_id=f"new{i}",
            canonical_fragment=f"new-{i}", atom_count=1)) for i in range(1600)])]
        combined = retain_train_pool(old, states)
        self.assertEqual(combined[:2607], before)
        self.assertEqual(len(combined), 4096)
        self.assertEqual(old, before)

    def test_pass_resume_and_uncommitted_attempt_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            contract = dict(self_sha256="contract", oracle_binding="gin", search_budget=asdict(SearchBudget()))
            info = dict(before=self.before, cache={}, source_unit_sha256="unit")
            fake = dict(pass_new_queries=128, new_graph_oracle_queries=128, parent_id="train1",
                witnesses=[], records=[], before=self.before)
            with patch.object(leaf, "search_parent", return_value=fake) as search:
                a = leaf._pass(target, contract, self.parent, info, predict=None, pool=[], maximum=128)
                b = leaf._pass(target, contract, self.parent, info, predict=None, pool=[], maximum=128)
                self.assertEqual(a, b)
                search.assert_called_once()
            saved = target / f"parent-{stable_sha256('train1')[:24]}-pass1.json"
            saved.unlink()  # fixture simulates missing committed result, leaves real intent
            with self.assertRaisesRegex(ValueError, "UNCOMMITTED_QUERY_PASS"):
                leaf._pass(target, contract, self.parent, info, predict=None, pool=[], maximum=128)

    def test_budget_excess_and_changed_previous_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = dict(self_sha256="contract", oracle_binding="gin", search_budget=asdict(SearchBudget()))
            info = dict(before=self.before, cache={}, source_unit_sha256="unit")
            with patch.object(leaf, "search_parent", return_value=dict(pass_new_queries=384, new_graph_oracle_queries=513)):
                with self.assertRaisesRegex(ValueError, "QUERY_BUDGET_EXCEEDED"):
                    leaf._pass(Path(tmp), contract, self.parent, info, predict=None, pool=[], maximum=384,
                        previous=dict(new_graph_oracle_queries=129))

    def test_no_gap_does_not_load_model_or_regenerate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            contract = dict(self_sha256="contract", eligible=[], previous_generation_queries=123)
            inputs = dict(output=target, source_freeze=dict(candidate_universe_sha256="old2607"))
            spec = dict(reach_candidate_file=str(target / "untouched.jsonl"))
            with patch.object(leaf, "prepare", return_value=(contract, inputs)), patch.object(leaf, "load_predictor") as load:
                result = leaf.run(spec, tmp)
                load.assert_not_called()
            self.assertEqual(result["state"], "NO_SUPPLEMENT_REQUIRED")
            self.assertEqual(result["candidate_universe_sha256"], "old2607")
            self.assertEqual(result["new_graph_oracle_queries"], 0)
            self.assertFalse((target / "candidate_universe.jsonl").exists())

    def test_full_train_terminal_required_before_parent_or_model_loading(self):
        spec = dict(output_root="/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments/"
                    "bace-gin-reach-aligned-v2/source", reach_search_contract="source-search")
        old = [dict(candidate_id=str(i)) for i in range(66)]
        pool = [dict(candidate_id=str(i)) for i in range(2607)]
        source = dict(spec_sha256=stable_sha256(spec))
        search = dict(search_budget=asdict(SearchBudget()), size_contract_source="actual-old-source",
                      test_opened=False)
        incomplete = dict(state="BOUNDED_PROBE_COMPLETE", spec_sha256=stable_sha256(spec),
                          parent_count=2, candidate_count=2607, test_loaded=False)
        with patch.object(leaf, "validate"), patch.object(leaf, "pools", return_value=(old, pool, {})), \
                patch.object(leaf, "verified", side_effect=[source, search, incomplete]), \
                patch.object(leaf.adapter, "fixed_source_parents") as parents:
            with self.assertRaisesRegex(ValueError, "COMPLETE_GIN_TRAIN2607_REQUIRED"):
                leaf.load_inputs(spec, spec["output_root"] + "-new-leaf")
            parents.assert_not_called()

    def test_two_pass_pipeline_uses_only_saved_train_and_no_repeated_queries_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            pool = [dict(candidate_id=f"r{i}", canonical_fragment=f"g{i}") for i in range(2607)]
            parents = [SimpleNamespace(parent_id=f"train{i}", smiles="CCCC", label=1) for i in range(2)]
            contract = dict(self_sha256="contract", eligible=[dict(index=i, parent_id=p.parent_id,
                unit_sha256=f"u{i}") for i, p in enumerate(parents)], previous_generation_queries=123,
                oracle_binding="gin", search_budget=asdict(SearchBudget()))
            inputs = dict(output=target, pool=pool, parents=parents, adoption=self.adoption)
            def inspect(_inputs, _spec, index):
                return None, dict(self_sha256=f"u{index}"), dict(before=self.before, cache={})
            calls = []
            def search(**kwargs):
                calls.append((kwargs["parent_id"], kwargs["maximum_new_queries"]))
                prior = kwargs["previous"] or {}
                return dict(parent_id=kwargs["parent_id"], before=kwargs["before"],
                    new_graph_oracle_queries=prior.get("new_graph_oracle_queries", 0)+kwargs["maximum_new_queries"],
                    pass_new_queries=kwargs["maximum_new_queries"], pass_cache_hits=2,
                    witnesses=[], records=[])
            with patch.object(leaf, "prepare", return_value=(contract, inputs)), \
                    patch.object(leaf, "_inspect", side_effect=inspect), \
                    patch.object(leaf, "load_predictor", return_value=lambda ss: []) as load, \
                    patch.object(leaf, "search_parent", side_effect=search):
                result = leaf.run({}, tmp)
                self.assertEqual(result["new_graph_oracle_queries"], 1024)
                self.assertEqual(result["parent_query_counts"], {"train0": 512, "train1": 512})
                self.assertEqual(result["candidate_count"], 2607)
                self.assertFalse(result["calibration_loaded"])
                self.assertFalse(result["test_loaded"])
                self.assertEqual(leaf.run({}, tmp), result)
                load.assert_called_once()
            self.assertEqual(Counter(calls), Counter({("train0", 128): 1, ("train1", 128): 1,
                ("train0", 384): 1, ("train1", 384): 1}))


if __name__ == "__main__":
    unittest.main()
