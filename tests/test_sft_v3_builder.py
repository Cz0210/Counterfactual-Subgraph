import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional local test dependency
    pd = None

from src.chem import is_rdkit_available
from src.data.hiv_dataset_utils import (
    HIVParentRecord,
    normalize_hiv_records,
    sample_records_by_strata,
)
from src.data.sft_v3_builder import (
    SFTV3BuilderConfig,
    SFTV3Example,
    label_stratified_scaffold_split,
    select_reference_candidate_for_parent,
    split_examples_scaffold_aware,
)


def _make_fake_example(
    sample_id: str,
    *,
    label: int,
    scaffold: str | None,
) -> SFTV3Example:
    normalized_scaffold = scaffold if scaffold is not None else ""
    return SFTV3Example(
        sample_id=sample_id,
        graph_id=sample_id,
        parent_smiles=f"C{sample_id}",
        label=label,
        parent_atom_count=12,
        scaffold_smiles=normalized_scaffold,
        instruction=f"prompt:{sample_id}",
        output="CCO",
        meta={
            "atom_ratio": 0.25,
            "candidate_strategy": "fg_carboxyl",
        },
    )


class SFTV3SplitTests(unittest.TestCase):
    def test_sft_v3_example_to_json_includes_completion_alias(self) -> None:
        example = _make_fake_example("sample", label=1, scaffold="scaf")

        payload = example.to_json()

        self.assertEqual(payload["prompt"], example.instruction)
        self.assertEqual(payload["response"], example.output)
        self.assertEqual(payload["completion"], "\nCCO")

    def test_label_stratified_scaffold_split_preserves_approximate_two_to_one_ratio(self) -> None:
        examples = [
            _make_fake_example(
                f"neg_{index}_{copy_index}",
                label=0,
                scaffold=f"neg_scaf_{index}",
            )
            for index in range(45)
            for copy_index in range(2)
        ] + [
            _make_fake_example(
                f"pos_{index}",
                label=1,
                scaffold=f"pos_scaf_{index}",
            )
            for index in range(45)
        ]

        train_examples, val_examples, summary = label_stratified_scaffold_split(
            examples,
            val_ratio=0.1,
            seed=7,
        )

        self.assertEqual(len(train_examples) + len(val_examples), len(examples))
        self.assertEqual(summary["scaffold_overlap_count"], 0)
        self.assertEqual(summary["split_method"], "label_stratified_scaffold")
        self.assertEqual(summary["target_val_total"], 14)
        self.assertIn(summary["val_label_counts"]["0"], {8, 9, 10})
        self.assertIn(summary["val_label_counts"]["1"], {4, 5, 6})
        self.assertLessEqual(abs(summary["label_val_target_error"]["0"]), 1)
        self.assertLessEqual(abs(summary["label_val_target_error"]["1"]), 1)

    def test_label_stratified_scaffold_split_keeps_scaffolds_disjoint(self) -> None:
        examples = [
            _make_fake_example(f"neg_{index}", label=0, scaffold=f"shared_neg_{index}")
            for index in range(30)
        ] + [
            _make_fake_example(f"pos_{index}", label=1, scaffold=f"shared_pos_{index}")
            for index in range(15)
        ]

        train_examples, val_examples, summary = label_stratified_scaffold_split(
            examples,
            val_ratio=0.2,
            seed=11,
        )

        train_scaffolds = {example.scaffold_smiles for example in train_examples}
        val_scaffolds = {example.scaffold_smiles for example in val_examples}
        self.assertTrue(train_examples)
        self.assertTrue(val_examples)
        self.assertEqual(summary["scaffold_overlap_count"], 0)
        self.assertFalse(train_scaffolds & val_scaffolds)

    def test_label_stratified_scaffold_split_handles_missing_scaffolds_without_one_huge_group(self) -> None:
        examples = [
            _make_fake_example(
                f"neg_no_scaf_{index}",
                label=0,
                scaffold="" if index % 2 == 0 else "ACYCLIC",
            )
            for index in range(40)
        ] + [
            _make_fake_example(
                f"pos_no_scaf_{index}",
                label=1,
                scaffold=None if index % 2 == 0 else "ACYCLIC",
            )
            for index in range(20)
        ]

        train_examples, val_examples, summary = label_stratified_scaffold_split(
            examples,
            val_ratio=0.1,
            seed=19,
        )

        self.assertEqual(len(train_examples) + len(val_examples), len(examples))
        self.assertEqual(summary["scaffold_overlap_count"], 0)
        self.assertGreaterEqual(summary["val_label_counts"]["0"], 3)
        self.assertGreaterEqual(summary["val_label_counts"]["1"], 1)
        self.assertGreater(summary["val_unique_scaffolds"], 1)
        self.assertLessEqual(abs(summary["label_val_target_error"]["0"]), 1)
        self.assertLessEqual(abs(summary["label_val_target_error"]["1"]), 1)

    def test_label_stratified_scaffold_split_records_target_error_with_large_group_pressure(self) -> None:
        examples = [
            _make_fake_example(
                f"neg_large_{index}",
                label=0,
                scaffold=f"neg_scaf_{index}",
            )
            for index in range(40)
        ] + [
            _make_fake_example(
                f"pos_big_{index}",
                label=1,
                scaffold="pos_big_group",
            )
            for index in range(8)
        ] + [
            _make_fake_example(
                f"pos_small_{index}",
                label=1,
                scaffold=f"pos_small_{index}",
            )
            for index in range(12)
        ]

        _train_examples, val_examples, summary = label_stratified_scaffold_split(
            examples,
            val_ratio=0.1,
            seed=23,
        )

        self.assertTrue(val_examples)
        self.assertIn("0", summary["label_val_target_error"])
        self.assertIn("1", summary["label_val_target_error"])
        self.assertEqual(summary["scaffold_overlap_count"], 0)
        self.assertGreater(summary["val_label_counts"]["0"], 0)
        self.assertGreater(summary["val_label_counts"]["1"], 0)
        self.assertLess(summary["val_label_counts"]["1"], len(val_examples))


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

    def test_normalize_hiv_records_supports_class_alias_and_string_positive_label(self) -> None:
        dataframe = pd.DataFrame(
            [
                {"smiles": "CCO", "class": "active"},
                {"smiles": "c1ccccc1O", "class": "inactive"},
            ]
        )

        records, summary = normalize_hiv_records(dataframe, positive_label="active")

        self.assertEqual(summary["smiles_column"], "smiles")
        self.assertEqual(summary["label_column"], "class")
        self.assertEqual(summary["positive_label_normalized"], "active")
        self.assertEqual(summary["negative_label_normalized"], "inactive")
        self.assertEqual(summary["valid_label_counts"]["1"], 1)
        self.assertEqual(summary["valid_label_counts"]["0"], 1)
        self.assertEqual([record.label for record in records], [1, 0])

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
        self.assertEqual(summary["split_method"], "label_stratified_scaffold")


if __name__ == "__main__":
    unittest.main()
