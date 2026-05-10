import unittest

from src.data.sft_column_compat import normalize_sft_example, validate_required_sft_fields


class TrainSFTColumnCompatTests(unittest.TestCase):
    def test_instruction_output_examples_gain_prompt_and_completion(self) -> None:
        example = {"instruction": "Generate...", "output": "CCO"}

        normalized = normalize_sft_example(example)

        self.assertEqual(normalized["instruction"], "Generate...")
        self.assertEqual(normalized["output"], "CCO")
        self.assertEqual(normalized["prompt"], "Generate...")
        self.assertEqual(normalized["completion"], "\nCCO")

    def test_prompt_completion_examples_are_preserved(self) -> None:
        example = {"prompt": "abc", "completion": "def"}

        normalized = normalize_sft_example(example)

        self.assertEqual(normalized["prompt"], "abc")
        self.assertEqual(normalized["completion"], "def")

    def test_instruction_input_output_examples_append_input_to_prompt(self) -> None:
        example = {
            "instruction": "Generate...",
            "input": "SMILES: CCO",
            "output": "CCO",
        }

        normalized = normalize_sft_example(example)

        self.assertEqual(normalized["prompt"], "Generate...\nSMILES: CCO")
        self.assertEqual(normalized["completion"], "\nCCO")

    def test_prompt_response_examples_gain_completion_alias(self) -> None:
        example = {"prompt": "abc", "response": "CCO"}

        normalized = normalize_sft_example(example)

        self.assertEqual(normalized["prompt"], "abc")
        self.assertEqual(normalized["response"], "CCO")
        self.assertEqual(normalized["completion"], "\nCCO")

    def test_missing_completion_raises_clear_value_error(self) -> None:
        normalized = normalize_sft_example({"instruction": "Generate..."})

        with self.assertRaisesRegex(
            ValueError,
            "missing columns=\\['completion'\\].*available columns=\\['instruction'\\]",
        ):
            validate_required_sft_fields(
                normalized,
                available_columns=["instruction"],
                split_name="train",
                row_index=0,
            )


if __name__ == "__main__":
    unittest.main()
