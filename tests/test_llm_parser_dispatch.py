"""Corrective transform tests, with prior source validator isolated as a seam."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ablations.llm import stage_dispatch_binding as binding


class ParserDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "new-driver"
        self.root.mkdir()
        self.commit = "b" * 40
        self.source = {"execution_commit": "a" * 40,
            "resource_only_overlay": {"owner_driver_root": str(self.base / "prior-driver")},
            "command": ["python", "-I", "-B", "/old/generation.py"],
            "variant_order": binding.ORDER, "downstream_commands": {}}
        self.policy = self.write("policy.json", {"old_bound_policy": True})
        self.source["resource_config"] = self.write("resource.json", {"memory": 100, "stage_file_policy": self.policy})
        self.repair = {"reason": "CONFIRMED_SHARED_SMILES_EXTRACTION_BUG", "prior_test_evaluated": True,
            "repair_selected_using_test": False, "model_generation_rerun": False,
            "gpu_science_allowed": False, "main_matrix_write": False,
            "output_root": str(self.base / "llm/corrective"), "registry_root": str(self.base / "llm/new-registry"),
            "owner_output_root": str(self.base / "single-owner"), "source_evaluation_audits": {}}
        for variant in binding.ORDER:
            output = self.base / "llm/old/evaluation" / variant
            candidate = self.base / "llm/old/generation" / variant
            task = {"calls": [{"parent": "train-one"}], "variant": variant}
            ref = self.write(variant + ".task.json", task)
            self.write(str(candidate / "candidate_generation_receipt.json"), {
                "status": "CANDIDATE_POOL_PASS", "variant": variant,
                "spec_sha256": binding.canonical_json_sha256(task), "next_call": 1})
            audit = self.write(str(output / "final_audit.json"), {"state": "PASS", "main_matrix_write": False})
            self.repair["source_evaluation_audits"][variant] = audit
            self.source["downstream_commands"][variant] = {"candidate_root": str(candidate), "output_root": str(output),
                "task_spec": ref, "command": ["python", "-I", "-B", "/old/evaluation.py", "--device", "cpu",
                    "--output-root", str(output), "--registry-root", str(self.base / "llm/old-registry"),
                    "--stage-file-policy", self.policy["path"], "--stage-file-policy-sha256", self.policy["sha256"],
                    "--compact-node-cache"]}
        self.source_ref = self.write("source-dispatch.json", self.source)
        self.spec = self.make_spec()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, payload):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload).encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def make_spec(self):
        result = copy.deepcopy(self.source)
        result.pop("resource_only_overlay")
        result.update(execution_commit=self.commit,
            parser_correction_overlay={"schema_version": "bace_saved_parser_cpu_correction_v1",
                "source_dispatch": self.source_ref, "repair_contract": self.write("repair.json", self.repair),
                "stage_file_policy": self.policy},
            downstream_commands=binding.parser_correction_commands(self.source, self.root, self.policy, self.repair))
        result["self_sha256"] = binding.canonical_json_sha256({k: v for k, v in result.items() if k != "self_sha256"})
        return result

    def validate(self, spec=None):
        with patch.object(binding, "validate_dispatch_runtime", return_value={}) as previous:
            result = binding._validate_parser_correction(spec or self.spec, self.commit, self.root)
            previous.assert_called_once_with(self.source, "a" * 40, str(self.base / "prior-driver"))
            return result

    def test_exact_transform_keeps_generation_and_cpu_only(self):
        receipt = self.validate()
        self.assertTrue(receipt["cpu_evaluation_only"])
        self.assertEqual(self.spec["command"], self.source["command"])
        self.assertEqual(receipt["owner_output_root"], str((self.base / "single-owner").resolve()))
        for row in self.spec["downstream_commands"].values():
            self.assertEqual(row["command"][row["command"].index("--device") + 1], "cpu")
            self.assertIn("--reparse-saved-raw", row["command"])

    def test_changed_command_or_gpu_rejected(self):
        bad = copy.deepcopy(self.spec)
        row = bad["downstream_commands"][binding.ORDER[0]]["command"]
        row[row.index("--device") + 1] = "cuda:0"
        with self.assertRaisesRegex(ValueError, "UNAUTHORIZED_DRIFT"):
            self.validate(bad)

    def test_original_registry_cannot_be_reused(self):
        self.repair["registry_root"] = str(self.base / "llm/old-registry")
        self.spec = self.make_spec()
        with self.assertRaisesRegex(ValueError, "OLD_REGISTRY_PROTECTED"):
            self.validate()

    def test_main_scope_cannot_be_output(self):
        self.repair["output_root"] = str(self.base / "main-matrix")
        self.spec = self.make_spec()
        with self.assertRaisesRegex(ValueError, "OUTSIDE_LLM_SCOPE"):
            self.validate()

    def test_missing_generation_cannot_fall_back_to_gpu(self):
        path = Path(self.source["downstream_commands"][binding.ORDER[0]]["candidate_root"]) / "candidate_generation_receipt.json"
        path.write_text('{"status":"RUNNING"}')
        with self.assertRaisesRegex(ValueError, "COMPLETED_RAW_GENERATION"):
            self.validate()

    def test_unseen_test_claim_rejected(self):
        self.repair["prior_test_evaluated"] = False
        self.spec = self.make_spec()
        with self.assertRaisesRegex(ValueError, "CORRECTION_SCOPE"):
            self.validate()

    def test_relative_output_rejected_before_resolution(self):
        self.repair["output_root"] = "relative-corrective"
        self.spec = self.make_spec()
        with self.assertRaisesRegex(ValueError, "ABSOLUTE_OUTPUT_REQUIRED"):
            self.validate()

    def test_queue_missing_completed_pool_cannot_be_false_all_pass(self):
        from src.ablations.llm.existing_gpu_owner import next_completed_evaluation
        path = Path(self.source["downstream_commands"][binding.ORDER[0]]["candidate_root"]) / "candidate_generation_receipt.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "GENERATION_DISAPPEARED"):
            next_completed_evaluation(self.spec)

    def test_queue_nonpass_pool_cannot_be_false_all_pass(self):
        from src.ablations.llm.existing_gpu_owner import next_completed_evaluation
        path = Path(self.source["downstream_commands"][binding.ORDER[0]]["candidate_root"]) / "candidate_generation_receipt.json"
        path.write_text('{"status":"RUNNING"}')
        with self.assertRaisesRegex(ValueError, "GENERATION_NOT_PASS"):
            next_completed_evaluation(self.spec)


if __name__ == "__main__":
    unittest.main()
