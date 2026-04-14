import json
import subprocess
import sys
import unittest

from src.chem import is_rdkit_available
from src.eval.inference import run_minimal_inference
from src.utils.paths import get_repo_root


class MinimalInferenceTests(unittest.TestCase):
    def test_run_minimal_inference_returns_structured_result(self) -> None:
        result = run_minimal_inference("CCO", label=1)

        self.assertEqual(result["mode"], "minimal_heuristic_inference")
        self.assertEqual(result["parent_smiles"], "CCO")
        self.assertTrue(result["fragment_candidate"])
        self.assertIn("checks", result)
        self.assertIn("validation", result["checks"])

        if is_rdkit_available():
            self.assertTrue(result["checks"]["validation"]["parseable"])
            self.assertTrue(result["checks"]["validation"]["chemically_valid"])
            self.assertTrue(result["checks"]["connected"])
            self.assertTrue(result["checks"]["is_substructure"])
        else:
            self.assertEqual(
                result["checks"]["fragment_parse"]["failure_type"],
                "rdkit_unavailable",
            )

    def test_run_infer_script_prints_json_result(self) -> None:
        repo_root = get_repo_root()
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_infer.py"),
            "--smiles",
            "CCO",
            "--label",
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        result = json.loads(completed.stdout)
        self.assertEqual(result["parent_smiles"], "CCO")
        self.assertIn("fragment_candidate", result)
        self.assertIn("checks", result)
        self.assertIn("run", result)
        self.assertIn(result["mode"], {"chemllm_local_inference", "heuristic_fallback_inference", "minimal_heuristic_inference"})


if __name__ == "__main__":
    unittest.main()
