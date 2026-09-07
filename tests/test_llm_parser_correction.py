import copy
import unittest
from src.models.llm_generator import clean_generated_smiles
from src.ablations.llm.parser_correction import reparse_attempts, reparse_scored_diagnostic


class StructuralExtractionTests(unittest.TestCase):
    def test_bracket_atom_map_colon_preserved(self):
        self.assertEqual(clean_generated_smiles("I would conclude that [O-:1]"), "[O-:1]")
        self.assertEqual(clean_generated_smiles("[CH3:1][OH:2]"), "[CH3:1][OH:2]")

    def test_aromatic_bond_colon_preserved(self):
        self.assertEqual(clean_generated_smiles("c1ccccc1:c1ccccc1"), "c1ccccc1:c1ccccc1")

    def test_json_smiles_and_native_quote(self):
        self.assertEqual(clean_generated_smiles('{"fragment_smiles":"[O-:1]"}'), "[O-:1]")
        self.assertEqual(clean_generated_smiles('```json\n{"smiles":"C:C"}\n```'), "C:C")
        self.assertEqual(clean_generated_smiles('"CO"'), "CO")

    def test_prose_pronoun_is_not_iodine(self):
        for text in ("I believe this is impossible", "I would conclude there is no answer", "ANSWER: I believe there is none"):
            self.assertEqual(clean_generated_smiles(text), "")
        self.assertEqual(clean_generated_smiles("I"), "I")
        self.assertEqual(clean_generated_smiles("SMILES: I"), "I")

    def test_no_chemistry_ranking_or_ring_repair(self):
        self.assertEqual(clean_generated_smiles("Candidate [CH2:1 followed by [O-:2]"), "[CH2:1")
        self.assertEqual(clean_generated_smiles("C1CC"), "C1CC")
        self.assertEqual(clean_generated_smiles("[C][Branch1][C]"), "[C][Branch1][C]")
        self.assertEqual(clean_generated_smiles("C.C"), "C.C")

    def test_no_prose_first_word_fallback(self):
        self.assertEqual(clean_generated_smiles("Unfortunately no answer is available"), "")
        self.assertEqual(clean_generated_smiles("The solution is [NH2:2]"), "[NH2:2]")
        self.assertEqual(clean_generated_smiles("My response is CCO"), "CCO")
        self.assertEqual(clean_generated_smiles("I believe the answer is O"), "O")

    def test_actual_project_wrapper_and_labeled_explanation(self):
        self.assertEqual(clean_generated_smiles("COUNTERFACTUAL_FRAGMENT_SMILES: CCO"), "CCO")
        self.assertEqual(clean_generated_smiles("COUNTERFACTUAL_FRAGMENT_SMILES: I"), "I")
        self.assertEqual(clean_generated_smiles("SMILES: CCO followed by explanation"), "CCO")
        self.assertEqual(clean_generated_smiles("ANSWER: I believe none"), "")

    def test_dot_components_are_preserved_not_repaired(self):
        self.assertEqual(clean_generated_smiles("Here is C.C"), "C.C")
        self.assertEqual(clean_generated_smiles("C.C."), "C.C.")
        self.assertEqual(clean_generated_smiles("SMILES: C.C."), "C.C.")

    def test_saved_attempts_unchanged_except_extraction(self):
        rows = [{"parent_id":"p", "attempt_index":0,"raw_text":"I believe [O-:1]",
                 "fragment_smiles":"1]", "train_only":True,"seed":7}]
        original=copy.deepcopy(rows)
        corrected,receipt=reparse_attempts(rows)
        self.assertEqual(rows,original)
        self.assertEqual(corrected,[{**original[0],"fragment_smiles":"[O-:1]"}])
        self.assertEqual(receipt["changed_attempts"],1)
        self.assertFalse(receipt["candidate_generation_rerun"])
        self.assertFalse(receipt["test_used_for_extraction"])

    def test_saved_attempt_missing_raw_rejected(self):
        with self.assertRaises(ValueError):
            reparse_attempts([{"train_only":True,"fragment_smiles":"O"}])

    def test_scored_diagnostic_rejects_test_or_duplicate(self):
        row={"stage":"LLM_COMMON_TRAIN_ONLY","test_loaded":False,"calibration_loaded":False,
             "parent_id":"p","candidate_index":0,"raw_output":"I believe none","raw_fragment":"I"}
        report=reparse_scored_diagnostic([row])
        self.assertEqual(report["prose_iodine_removed_count"],1)
        self.assertFalse(report["scientific_evaluation_complete"])
        with self.assertRaises(ValueError):reparse_scored_diagnostic([row,row])
        with self.assertRaises(ValueError):reparse_scored_diagnostic([{**row,"test_loaded":True}])


if __name__ == "__main__":
    unittest.main()
