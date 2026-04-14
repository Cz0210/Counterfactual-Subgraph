import unittest

from src.chem import (
    ChemistryFailureType,
    delete_fragment_from_parent,
    is_connected_fragment,
    is_parent_substructure,
    is_rdkit_available,
    parse_smiles,
    validate_fragment_candidate,
)


class ChemistrySmokeTests(unittest.TestCase):
    def test_parse_smiles_reports_empty_input(self) -> None:
        parsed = parse_smiles("   ")

        self.assertFalse(parsed.parseable)
        self.assertEqual(parsed.failure_type, ChemistryFailureType.EMPTY_SMILES)

    def test_parse_smiles_handles_rdkit_availability(self) -> None:
        parsed = parse_smiles("CCO")

        if is_rdkit_available():
            self.assertTrue(parsed.parseable)
            self.assertTrue(parsed.sanitized)
            self.assertEqual(parsed.canonical_smiles, "CCO")
            self.assertEqual(parsed.atom_count, 3)
        else:
            self.assertFalse(parsed.parseable)
            self.assertEqual(parsed.failure_type, ChemistryFailureType.RDKIT_UNAVAILABLE)

    def test_connectedness_and_substructure_contract(self) -> None:
        if not is_rdkit_available():
            self.assertFalse(is_connected_fragment("CCO"))
            self.assertFalse(is_parent_substructure("CCO", "CO"))
            return

        self.assertTrue(is_connected_fragment("CCO"))
        self.assertFalse(is_connected_fragment("C.C"))
        self.assertTrue(is_parent_substructure("CCO", "CO"))
        self.assertFalse(is_parent_substructure("CCO", "N#N"))

    def test_deletion_contract_returns_clear_result(self) -> None:
        result = delete_fragment_from_parent("CCO", "CO")

        if is_rdkit_available():
            self.assertTrue(result.success)
            self.assertEqual(result.residual_smiles, "C")
            self.assertEqual(result.match_count, 1)
            self.assertTrue(result.selected_match)
        else:
            self.assertFalse(result.success)
            self.assertEqual(result.failure_type, ChemistryFailureType.RDKIT_UNAVAILABLE)

    def test_validation_surfaces_disconnected_fragment_failure(self) -> None:
        result = validate_fragment_candidate("CCO", "C.C")

        if is_rdkit_available():
            self.assertFalse(result.connected)
            self.assertIn(ChemistryFailureType.DISCONNECTED_FRAGMENT, result.failure_types)
        else:
            self.assertFalse(result.chemically_valid)
            self.assertIn(ChemistryFailureType.RDKIT_UNAVAILABLE, result.failure_types)


if __name__ == "__main__":
    unittest.main()
