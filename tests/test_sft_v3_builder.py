import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional local test dependency
    pd = None

from src.chem import is_rdkit_available
from src.data.hiv_dataset_utils import HIVParentRecord, normalize_hiv_records, sample_records_by_strata
from src.data.sft_v3_builder import (
    SFTV3BuilderConfig,
    SFTV3Example,
    select_reference_candidate_for_parent,
    split_examples_scaffold_aware,
)


@unittest.skipUnless(
    is_rdkit_available() and pd is not None,
    "RDKit and pandas are required for SFT v3 builder tests",
)
class SFTV3BuilderTests(unittest.TestCase):
    def test_normalize_hiv_records_detects_common_columns(self) -> None:
        dataframe = pd.DataFrame(
            [
                {"SMILES": "CCO", "HIV_active": 1},
                {"SMILES": "c1ccccc1O", "HIV_active": 0},
                {"SMILES": "", "HIV_active": 0},
            ]
        )

        records, summary = normalize_hiv_records(dataframe, positive_label=1)

        self.assertEqual(summary["smiles_column"], "SMILES")
        self.assertEqual(summary["label_column"], "HIV_active")
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["valid_label_counts"]["1"], 1)
        self.assertEqual(summary["valid_label_counts"]["0"], 1)
        self.assertGreaterEqual(summary["dropped_counts"]["empty_smiles"], 1)

    def test_sample_records_by_strata_spreads_across_groups(self) -> None:
        records = [
            HIVParentRecord("r0", 0, "C", "C", 0, 0, 10, "scaf_a", "atoms_00_15"),
            HIVParentRecord("r1", 1, "CC", "CC", 0, 0, 11, "scaf_a", "atoms_00_15"),
            HIVParentRecord("r2", 2, "CCC", "CCC", 0, 0, 12, "scaf_b", "atoms_00_15"),
            HIVParentRecord("r3", 3, "CCCC", "CCCC", 0, 0, 13, "scaf_c", "atoms_16_25"),
        ]

        sampled = sample_records_by_strata(records, sample_size=3, seed=7)

        self.assertEqual(len(sampled), 3)
        self.assertGreaterEqual(len({record.stratum_key for record in sampled}), 2)

    def test_select_reference_candidate_for_parent_returns_mid_size_candidate(self) -> None:
        record = HIVParentRecord(
            sample_id="aspirin",
            source_row_index=0,
            source_smiles="CC(=O)Oc1ccccc1C(=O)O",
            parent_smiles="CC(=O)Oc1ccccc1C(=O)O",
            label=1,
            raw_label=1,
            parent_atom_count=13,
            scaffold_smiles="c1ccccc1",
            size_bin="atoms_00_15",
        )
        config = SFTV3BuilderConfig()

        result = select_reference_candidate_for_parent(record, config=config)

        self.assertIsNotNone(result.selected_candidate)
        candidate = result.selected_candidate
        assert candidate is not None
        self.assertTrue(candidate.residual_nonempty)
        self.assertGreaterEqual(candidate.atom_ratio, config.min_atom_ratio)
        self.assertLessEqual(candidate.atom_ratio, config.max_atom_ratio)
        self.assertNotIn("*", candidate.core_fragment)
        self.assertTrue(candidate.raw_fragment)

    def test_split_examples_scaffold_aware_avoids_overlap_when_possible(self) -> None:
        examples = [
            SFTV3Example(
                sample_id=f"id_{index}",
                graph_id=f"id_{index}",
                parent_smiles=f"C{index}",
                label=index % 2,
                parent_atom_count=10 + index,
                scaffold_smiles=f"scaf_{index}",
                instruction=f"prompt_{index}",
                output="CCO",
                meta={"atom_ratio": 0.2 + (0.05 * index), "candidate_strategy": "fg_carboxyl"},
            )
            for index in range(6)
        ]

        train_examples, val_examples, summary = split_examples_scaffold_aware(
            examples,
            val_ratio=0.33,
            seed=7,
        )

        self.assertTrue(train_examples)
        self.assertTrue(val_examples)
        self.assertEqual(summary["scaffold_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
