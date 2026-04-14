import unittest

from src.data import AIDSHIVCsvDataset, sample_random_aids_hiv_record
from src.utils.paths import get_repo_root


class AIDSHIVDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = get_repo_root()
        self.dataset_path = self.repo_root / "data" / "raw" / "AIDS" / "HIV.csv"

    def test_dataset_loads_non_empty_records(self) -> None:
        dataset = AIDSHIVCsvDataset.from_csv(self.dataset_path)

        self.assertGreater(len(dataset), 0)
        self.assertTrue(dataset[0].smiles)
        self.assertIn(dataset[0].label, (0, 1))

    def test_sampling_is_seed_stable(self) -> None:
        sample_a = sample_random_aids_hiv_record(self.dataset_path, seed=7)
        sample_b = sample_random_aids_hiv_record(self.dataset_path, seed=7)

        self.assertEqual(sample_a.smiles, sample_b.smiles)
        self.assertEqual(sample_a.label, sample_b.label)


if __name__ == "__main__":
    unittest.main()
