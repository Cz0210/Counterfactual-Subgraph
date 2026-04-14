import unittest

from src.chem import (
    ChemistryFailureType,
    delete_fragment_from_parent,
    get_remainder_graph,
    is_connected_fragment,
    is_rdkit_available,
    is_valid_capped_subgraph,
    parse_smiles,
    validate_fragment_candidate,
)


class CappedChemistryTests(unittest.TestCase):
    def test_parse_capped_smiles_contract(self) -> None:
        parsed = parse_smiles("*c1ccccc1")

        if is_rdkit_available():
            self.assertTrue(parsed.parseable)
            self.assertTrue(parsed.sanitized)
            self.assertTrue(parsed.contains_dummy_atoms)
        else:
            self.assertFalse(parsed.parseable)
            self.assertEqual(parsed.failure_type, ChemistryFailureType.RDKIT_UNAVAILABLE)

    def test_connected_capped_fragment_contract(self) -> None:
        if not is_rdkit_available():
            self.assertFalse(is_connected_fragment("*c1ccccc1"))
            return

        self.assertTrue(is_connected_fragment("*c1ccccc1"))

    def test_capped_subgraph_requires_boundary_coverage(self) -> None:
        if not is_rdkit_available():
            self.assertFalse(is_valid_capped_subgraph("Cc1ccccc1", "*c1ccccc1"))
            return

        self.assertTrue(is_valid_capped_subgraph("Cc1ccccc1", "*c1ccccc1"))
        self.assertFalse(is_valid_capped_subgraph("Cc1ccccc1", "c1ccccc1"))

    def test_capped_deletion_returns_capped_remainder(self) -> None:
        result = delete_fragment_from_parent("Cc1ccccc1", "*c1ccccc1")

        if not is_rdkit_available():
            self.assertFalse(result.success)
            self.assertEqual(result.failure_type, ChemistryFailureType.RDKIT_UNAVAILABLE)
            return

        self.assertTrue(result.success)
        self.assertEqual(result.residual_smiles, "*C")
        self.assertEqual(get_remainder_graph("Cc1ccccc1", "*c1ccccc1"), "*C")

    def test_validation_reports_capped_subgraph_status(self) -> None:
        result = validate_fragment_candidate("Cc1ccccc1", "*c1ccccc1")

        if not is_rdkit_available():
            self.assertFalse(result.chemically_valid)
            self.assertIn(ChemistryFailureType.RDKIT_UNAVAILABLE, result.failure_types)
            return

        self.assertTrue(result.is_substructure)
        self.assertTrue(result.deletion_supported)
        self.assertEqual(result.residual_smiles, "*C")


if __name__ == "__main__":
    unittest.main()
