import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.train_aids_oracle import _load_dataset, pd


@unittest.skipIf(pd is None, "pandas is required for CSV loading tests")
class TrainAidsOracleDatasetTests(unittest.TestCase):
    def test_load_dataset_prefers_hiv_active_column(self) -> None:
        csv_content = "\n".join(
            [
                "smiles,activity,HIV_active",
                "CCO,CI,1",
                "CCC,CA,0",
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "hiv_active_first.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            frame = _load_dataset(csv_path)

        self.assertEqual(frame["smiles"].tolist(), ["CCO", "CCC"])
        self.assertEqual(frame["activity"].tolist(), [1, 0])

    def test_load_dataset_maps_nci_activity_strings(self) -> None:
        csv_content = "\n".join(
            [
                "smiles,activity",
                "CCO,CI",
                "CCC,CM",
                "CCN,CA",
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "nci_activity.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            frame = _load_dataset(csv_path)

        self.assertEqual(frame["smiles"].tolist(), ["CCO", "CCC", "CCN"])
        self.assertEqual(frame["activity"].tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
