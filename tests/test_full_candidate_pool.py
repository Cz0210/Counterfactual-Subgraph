from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.eval.full_candidate_pool import (
    FullPoolGenerationConfig,
    build_generation_kwargs,
    generate_ids_with_sanitized_kwargs,
    inspect_checkpoint_directory,
    resolve_adapter_load_path,
    set_global_generation_seed,
)


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

    def test_build_generation_kwargs_does_not_include_generator(self) -> None:
        config = FullPoolGenerationConfig(
            num_return_sequences=4,
            generation_temperature=0.7,
            generation_top_p=0.9,
            generation_do_sample=True,
            max_new_tokens=96,
        )
        encoded = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "generator": object(),
        }

        class _Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

        generation_kwargs = build_generation_kwargs(
            encoded=encoded,
            tokenizer=_Tokenizer(),
            config=config,
        )

        self.assertNotIn("generator", generation_kwargs)
        self.assertEqual(generation_kwargs["max_new_tokens"], 96)
        self.assertEqual(generation_kwargs["num_return_sequences"], 4)
        self.assertTrue(generation_kwargs["do_sample"])

    def test_set_global_generation_seed_calls_all_seed_hooks(self) -> None:
        fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=mock.Mock()))
        fake_cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            manual_seed_all=mock.Mock(),
        )
        fake_torch = SimpleNamespace(
            manual_seed=mock.Mock(),
            cuda=fake_cuda,
        )
        fake_transformers = SimpleNamespace(set_seed=mock.Mock())

        with mock.patch("random.seed") as random_seed:
            with mock.patch.dict(
                sys.modules,
                {
                    "numpy": fake_numpy,
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                },
            ):
                set_global_generation_seed(13)

        random_seed.assert_called_once_with(13)
        fake_numpy.random.seed.assert_called_once_with(13)
        fake_torch.manual_seed.assert_called_once_with(13)
        fake_torch.cuda.is_available.assert_called_once_with()
        fake_torch.cuda.manual_seed_all.assert_called_once_with(13)
        fake_transformers.set_seed.assert_called_once_with(13)

    def test_generate_ids_with_sanitized_kwargs_never_passes_generator(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class _DummyModel:
            def generate(self, **kwargs):  # type: ignore[no-untyped-def]
                captured_kwargs.update(kwargs)
                return [[1, 2, 3]]

        class _NoGradContext:
            def __enter__(self) -> None:
                return None

            def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
                return None

        class _FakeTorchModule:
            @staticmethod
            def no_grad() -> _NoGradContext:
                return _NoGradContext()

        generation_kwargs = {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "generator": object(),
            "max_new_tokens": 16,
        }

        output = generate_ids_with_sanitized_kwargs(
            _DummyModel(),
            generation_kwargs,
            torch_module=_FakeTorchModule(),
        )

        self.assertNotIn("generator", captured_kwargs)
        self.assertEqual(output, [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
