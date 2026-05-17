from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.eval.full_candidate_pool import inspect_checkpoint_directory, resolve_adapter_load_path


class FullCandidatePoolCheckpointTests(unittest.TestCase):
    def test_inspect_checkpoint_directory_prefers_root_adapter_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ppo_root"
            ckpt_dir.mkdir()
            (ckpt_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (ckpt_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
            (ckpt_dir / "candidate_pool.jsonl").write_text("", encoding="utf-8")
            (ckpt_dir / "trainer_state.json").write_text("{}", encoding="utf-8")
            (ckpt_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (ckpt_dir / "decoded_chem_value_head.pt").write_text("stub", encoding="utf-8")

            inspection = inspect_checkpoint_directory(ckpt_dir)

            self.assertTrue(inspection.exists)
            self.assertTrue(inspection.root_has_adapter)
            self.assertEqual(inspection.selected_load_mode, "root_adapter")
            self.assertEqual(Path(inspection.selected_load_path), ckpt_dir.resolve())
            self.assertTrue(inspection.candidate_pool_exists)
            self.assertTrue(inspection.trainer_state_exists)

    def test_resolve_adapter_load_path_falls_back_to_latest_checkpoint_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ppo_nested"
            ckpt_dir.mkdir()
            checkpoint_100 = ckpt_dir / "checkpoint-100"
            checkpoint_100.mkdir()
            checkpoint_150 = ckpt_dir / "checkpoint-150"
            checkpoint_150.mkdir()
            (checkpoint_100 / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_150 / "adapter_config.json").write_text("{}", encoding="utf-8")

            resolved = resolve_adapter_load_path(ckpt_dir)

            self.assertEqual(resolved, checkpoint_150.resolve())


if __name__ == "__main__":
    unittest.main()
