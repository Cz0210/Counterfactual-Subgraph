import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.utils.env import (
    ExecutionEnvironment,
    apply_dotlist_overrides,
    detect_execution_environment,
    load_and_merge_config_files,
)
from src.utils.paths import build_runtime_paths, get_repo_root, inject_runtime_paths


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = get_repo_root()

    def test_load_and_merge_stage_configs(self) -> None:
        config = load_and_merge_config_files(
            [
                self.repo_root / "configs" / "base.yaml",
                self.repo_root / "configs" / "local.yaml",
                self.repo_root / "configs" / "sft.yaml",
            ]
        )

        self.assertEqual(config["runtime"]["environment"], "local")
        self.assertEqual(config["training"]["stage"], "weak_supervision_sft")
        self.assertEqual(config["run"]["name"], "sft")
        self.assertFalse(Path(config["paths"]["output_root"]).is_absolute())

    def test_dotlist_override_and_path_injection(self) -> None:
        config = load_and_merge_config_files([self.repo_root / "configs" / "base.yaml"])
        config = apply_dotlist_overrides(
            config,
            [
                "run.name=unit_test",
                "model.model_path=models/local_model",
                "model.tokenizer_path=models/local_tokenizer",
            ],
        )

        runtime_paths = build_runtime_paths(config, stage_name="sft", create_dirs=False)
        manifest = inject_runtime_paths(config, runtime_paths)

        self.assertTrue(manifest["resolved_paths"]["run_dir"].endswith("/sft/unit_test"))
        self.assertTrue(manifest["resolved_paths"]["model_path"].endswith("models/local_model"))
        self.assertTrue(
            manifest["resolved_paths"]["tokenizer_path"].endswith("models/local_tokenizer")
        )

    def test_detect_execution_environment_prefers_slurm(self) -> None:
        detected = detect_execution_environment({"SLURM_JOB_ID": "12345"})
        self.assertEqual(detected, ExecutionEnvironment.HPC)

    def test_run_sft_script_supports_config_and_cli(self) -> None:
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "run_sft.py"),
            "--run-name",
            "unit_cli",
            "--model-path",
            "models/local_model",
            "--tokenizer-path",
            "models/local_tokenizer",
            "--print-config",
        ]
        result = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["run"]["name"], "unit_cli")
        self.assertEqual(manifest["runtime"]["environment"], "local")
        self.assertTrue(manifest["resolved_paths"]["run_dir"].endswith("/sft/unit_cli"))


if __name__ == "__main__":
    unittest.main()
