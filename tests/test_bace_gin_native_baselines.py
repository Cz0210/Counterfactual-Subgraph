import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.bace_gin_native_baselines import (
    NATIVE_CONTRACTS, evaluate_parent, prepare_candidate_predictions, select_order, validate_original_pool,
)


BINDING = {"backbone": "gin", "model_sha256": "gin-model", "temperature_sha256": "fit-temp", "feature_schema_sha256": "schema"}


def candidate(identifier="r1", smiles="CO", method="gcfexplainer"):
    kind, semantics, _ = NATIVE_CONTRACTS[method]
    return {"candidate_id": identifier, "canonical_smiles": smiles,
            "action_kind": kind, "action_semantics": semantics, "generation_split": "train",
            "test_loaded": False, "candidate_set_preselected": False, "lineage_validated": True}


class Oracle:
    def __init__(self): self.calls = []
    def predict_records(self, graphs, batch_size):
        self.calls.append(list(graphs))
        return [{"logits": [2., 0.], "probabilities": [.9, .1], "predicted_label": 0} for _ in graphs]


class Distance:
    def __init__(self): self.calls = []
    def distance(self, parent, complete_graph):
        self.calls.append((parent, complete_graph))
        return {"ok": True, "distance": .2, "cache_hit": True}
    def distance_for_action(self, *args, **kwargs):
        raise AssertionError("Native full graphs must not fabricate deletion match keys")


class NativeGINAdapterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [candidate(), candidate("r2", "CO")]
        self.oracle, self.distance = Oracle(), Distance()
        self.cache = prepare_candidate_predictions(self.rows, self.oracle, None, oracle_binding=BINDING,
                            batch_size=1, graph_builder=lambda _, **kw: kw["smiles"])
        self.before = {"logits": [0., 2.], "probabilities": [.1, .9], "predicted_label": 1}

    def evaluate(self, parent="CC", before=None, rows=None, binding=None):
        return evaluate_parent(types.SimpleNamespace(parent_id="p", smiles=parent), rows or self.rows,
            self.oracle, None, self.distance, "calibration", before=before or self.before,
            method="gcfexplainer", oracle_binding=binding or BINDING, candidate_predictions=self.cache)

    def test_full_original_pool_not_selected20(self):
        with self.assertRaisesRegex(ValueError, "ORIGINAL_PRESELECT_POOL_COUNT"):
            validate_original_pool("gcfexplainer", [candidate(str(i)) for i in range(20)])
        validate_original_pool("gcfexplainer", self.rows, expected_count=2)

    def test_unique_candidate_gin_inference_once_across_parents(self):
        self.evaluate(); self.evaluate("CCC")
        self.assertEqual(self.oracle.calls, [["CO"]])
        self.assertEqual(self.cache["unique_graph_count"], 1)
        self.assertFalse(self.cache["parent_flip_masks_reused"])

    def test_native_fullgraph_not_deletion_and_positive_flip(self):
        pairs, apps = self.evaluate()
        self.assertTrue(pairs[0]["pair_strict_flip"])
        self.assertAlmostEqual(pairs[0]["cf_drop"], .8)
        self.assertEqual(pairs[0]["classifier_family"], "gin")
        self.assertEqual(self.distance.calls[0], ("CC", "CO"))
        self.assertIsNone(apps[0]["match_atom_indices"])
        self.assertIsNone(apps[0]["delete_valid"])
        self.assertFalse(apps[0]["operation_is_deletion"])

    def test_same_after_graph_does_not_reuse_parent_flip(self):
        pairs, _ = self.evaluate(before={"logits": [2., 0.], "probabilities": [.9, .1], "predicted_label": 0})
        self.assertFalse(pairs[0]["cf_flip"])
        self.assertIsNone(pairs[0]["wnode_distance"])
        self.assertEqual(self.distance.calls, [])

    def test_weight_temperature_schema_drift_rejected(self):
        for field in ("model_sha256", "temperature_sha256", "feature_schema_sha256"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "CACHE_BINDING_MISMATCH"):
                    self.evaluate(binding={**BINDING, field: "changed"})

    def test_id_or_graph_changed_rejected(self):
        for row in (candidate("foreign"), candidate(smiles="CCC")):
            with self.assertRaisesRegex(ValueError, "CACHE_CANDIDATE_IDENTITY_MISMATCH"):
                self.evaluate(rows=[row])

    def test_candidate_train_boundary_and_native_actions_preserved(self):
        for field, value in (("test_loaded", True), ("candidate_set_preselected", True),
                             ("generation_split", "test"), ("action_kind", "hard_deletion")):
            row = candidate(); row[field] = value
            with self.assertRaises(ValueError): validate_original_pool("gcfexplainer", [row], expected_count=1)
        row = candidate(method="comrecgc"); row["lineage_validated"] = False
        with self.assertRaisesRegex(ValueError, "LINEAGE"):
            validate_original_pool("comrecgc", [row], expected_count=1)

    def test_nonfinite_prediction_refused(self):
        self.cache = copy.deepcopy(self.cache)
        self.cache["predictions_by_smiles"]["CO"]["logits"] = [float("nan"), 0.]
        with self.assertRaisesRegex(ValueError, "NOT_FINITE"):
            self.evaluate()

    def test_failed_raw_cost_not_zero_or_flip(self):
        self.distance.distance = lambda *a: {"ok": False, "error": "CACHE_PROVENANCE_GAP"}
        pairs, apps = self.evaluate()
        self.assertTrue(pairs[0]["cf_flip"])
        self.assertFalse(pairs[0]["pair_strict_flip"])
        self.assertEqual(pairs[0]["failure_reason"], "CACHE_PROVENANCE_GAP")
        self.assertIsNone(pairs[0]["wnode_distance"])

    def test_original_selector_no_cap_no_reach_no_test_variant_choice(self):
        config = {"top_k": 20, "table_k": 10, "seed": 13, "local_swap_passes": 2,
                  "parent_limit": 0, "candidate_limit": 0, "forbid_test": True,
                  "prefix_weights": [1.] * 10 + [.5] * 10}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "variants" / "A1_SingleTheta").mkdir(parents=True)
            (root / "calibration_decision.json").write_text(json.dumps({
                "selected_variant": "A1_SingleTheta", "decision_rule": ["original calibration decision"]}))
            (root / "variants" / "A1_SingleTheta" / "selected_top20.json").write_text(json.dumps({"candidates": self.rows}))
            ctx = {"method": "gcfexplainer", "test_loaded": False, "original_selector_config": config,
                   "output_root": tmp, "thresholds": {"exact": "old-grid"}, "threshold_provenance": {"source": "original"}}
            with patch("src.eval.mutagenicity_wnode_selector.run_mutagenicity_wnode_selector") as run, \
                 patch("src.eval.mutagenicity_wnode_selector.threshold_bundle_from_dict", return_value="BUNDLE"):
                result = select_order("calibration", ctx)
                self.assertEqual(result["selected_variant"], "A1_SingleTheta")
                self.assertEqual(run.call_args.kwargs["candidate_limit"], 0)
                self.assertEqual(run.call_args.kwargs["local_swap_passes"], 2)
                self.assertEqual(run.call_args.kwargs["frozen_thresholds"], "BUNDLE")
                self.assertTrue(run.call_args.kwargs["forbid_test"])
                with self.assertRaisesRegex(ValueError, "CONFIG_CHANGED"):
                    select_order("calibration", {**ctx, "original_selector_config": {**config, "candidate_limit": 20}})
                with self.assertRaisesRegex(ValueError, "CALIBRATION_CONTEXT"):
                    select_order("calibration", {**ctx, "test_loaded": True})


if __name__ == "__main__": unittest.main()
