import unittest
from unittest.mock import Mock

import pandas as pd

from src.data.sft_preparation import (
    build_balanced_candidate_pool,
    build_sft_instruction,
    label_ratio,
    prepare_balanced_sft_examples,
    split_examples,
)


class PrepareSFTDataTests(unittest.TestCase):
    def test_instruction_matches_minimal_prompt_contract(self) -> None:
        instruction = build_sft_instruction("CCO")
        expected = (
            "[System]\n"
            "Generate a valid, chemically capped subgraph for the following parent molecule. "
            "Output only the fragment SMILES.\n\n"
            "[Input]\n"
            "PARENT_SMILES: CCO\n\n"
            "[Output]\n"
        )
        self.assertEqual(instruction, expected)

    def test_balanced_pool_keeps_all_positives_and_fills_with_negatives(self) -> None:
        valid_records = pd.DataFrame(
            [
                {"source_row_index": 0, "parent_smiles": "P0", "smiles_raw": "P0", "HIV_active": 1},
                {"source_row_index": 1, "parent_smiles": "P1", "smiles_raw": "P1", "HIV_active": 1},
                {"source_row_index": 2, "parent_smiles": "N0", "smiles_raw": "N0", "HIV_active": 0},
                {"source_row_index": 3, "parent_smiles": "N1", "smiles_raw": "N1", "HIV_active": 0},
                {"source_row_index": 4, "parent_smiles": "N2", "smiles_raw": "N2", "HIV_active": 0},
                {"source_row_index": 5, "parent_smiles": "N3", "smiles_raw": "N3", "HIV_active": 0},
                {"source_row_index": 6, "parent_smiles": "N4", "smiles_raw": "N4", "HIV_active": 0},
            ]
        )

        sampling = build_balanced_candidate_pool(valid_records, total_examples=5, seed=13)

        self.assertEqual(sampling.valid_positive_count, 2)
        self.assertEqual(sampling.valid_negative_count, 5)
        self.assertEqual(sampling.selected_positive_count, 2)
        self.assertEqual(sampling.selected_negative_count, 3)
        self.assertEqual(len(sampling.base_records), 5)
        self.assertEqual(len(sampling.refill_negative_records), 2)
        self.assertEqual(int((sampling.base_records["HIV_active"] == 1).sum()), 2)

    def test_prepare_examples_backfills_failed_base_records(self) -> None:
        valid_records = pd.DataFrame(
            [
                {"source_row_index": 0, "parent_smiles": "P0", "smiles_raw": "P0", "HIV_active": 1},
                {"source_row_index": 1, "parent_smiles": "P1", "smiles_raw": "P1", "HIV_active": 1},
                {"source_row_index": 2, "parent_smiles": "N0", "smiles_raw": "N0", "HIV_active": 0},
                {"source_row_index": 3, "parent_smiles": "N1", "smiles_raw": "N1", "HIV_active": 0},
                {"source_row_index": 4, "parent_smiles": "N2", "smiles_raw": "N2", "HIV_active": 0},
                {"source_row_index": 5, "parent_smiles": "N3", "smiles_raw": "N3", "HIV_active": 0},
                {"source_row_index": 6, "parent_smiles": "N4", "smiles_raw": "N4", "HIV_active": 0},
            ]
        )
        fragment_lookup = {
            "P0": "*CO",
            "P1": None,
            "N0": None,
            "N1": "*CCO",
            "N2": "*c1ccccc1",
            "N3": "*CN",
            "N4": "*CCC",
        }
        builder = Mock(side_effect=lambda parent_smiles, rng: fragment_lookup[parent_smiles])

        examples, summary = prepare_balanced_sft_examples(
            valid_records,
            total_examples=5,
            seed=7,
            show_progress=False,
            fragment_builder=builder,
        )

        self.assertEqual(len(examples), 5)
        self.assertEqual(summary.successful_examples, 5)
        self.assertGreaterEqual(summary.failed_fragment_records, 2)
        self.assertGreater(summary.refill_records_attempted, 0)
        self.assertEqual(sum(example.label == 1 for example in examples), 1)
        self.assertEqual(sum(example.label == 0 for example in examples), 4)

    def test_split_and_ratio_helpers_are_stable(self) -> None:
        examples = [
            type("Example", (), {"label": 1})(),
            type("Example", (), {"label": 1})(),
            type("Example", (), {"label": 0})(),
            type("Example", (), {"label": 0})(),
        ]
        ratio = label_ratio(examples)

        self.assertAlmostEqual(ratio[1], 0.5)
        self.assertAlmostEqual(ratio[0], 0.5)

        prepared = [
            type(
                "Prepared",
                (),
                {
                    "instruction": f"i{index}",
                    "output": f"o{index}",
                    "parent_smiles": f"p{index}",
                    "label": index % 2,
                },
            )()
            for index in range(4)
        ]
        train_examples, val_examples = split_examples(prepared, train_size=3, val_size=1, seed=11)
        self.assertEqual(len(train_examples), 3)
        self.assertEqual(len(val_examples), 1)


if __name__ == "__main__":
    unittest.main()
