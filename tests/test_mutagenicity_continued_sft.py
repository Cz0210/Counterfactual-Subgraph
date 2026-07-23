from __future__ import annotations

import csv
from pathlib import Path
import re

import pytest

pytest.importorskip("rdkit")

from scripts.train_mutagenicity_continued_sft import _nested, build_parser
from src.data.mutagenicity_continued_sft import (
    ContinuedSFTRecord,
    ParentCoverageTracker,
    build_checkpoint_manifest,
    deterministic_smoke_sample,
    ensure_new_output_root,
    load_continued_sft_records,
    tokenize_completion_only,
    validate_peft_checkpoint,
    validate_train_val_isolation,
)
from src.utils.env import load_and_merge_config_files


FIELDS = (
    "molecule_id",
    "parent_smiles",
    "label",
    "source_label",
    "target_label",
    "semantic_label",
    "split",
    "scaffold_smiles",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
    "prompt",
    "completion",
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [10 + (ord(character) % 71) for character in text]}


def _row(
    molecule_id: str,
    smiles: str,
    split: str,
    scaffold: str,
    **updates: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "molecule_id": molecule_id,
        "parent_smiles": smiles,
        "label": 1,
        "source_label": 1,
        "target_label": 0,
        "semantic_label": "mutagenic",
        "split": split,
        "scaffold_smiles": scaffold,
        "teacher_pred": 1,
        "teacher_prob_0": 0.1,
        "teacher_prob_1": 0.9,
        "teacher_correct": True,
        "prompt": f"SMILES: {smiles}\nFRAGMENT:",
        "completion": "\nCC",
    }
    row.update(updates)
    return row


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _record(
    index: int,
    *,
    split: str = "train",
    scaffold: str | None = None,
) -> ContinuedSFTRecord:
    smiles = "C" * (index + 2)
    return ContinuedSFTRecord(
        molecule_id=f"MUT_{split}_{index}",
        parent_smiles=smiles,
        canonical_parent_smiles=smiles,
        label=1,
        source_label=1,
        target_label=0,
        semantic_label="mutagenic",
        split=split,
        scaffold_smiles=scaffold or f"SCAF_{index % 3}",
        teacher_pred=1,
        teacher_prob_0=0.1,
        teacher_prob_1=0.9,
        teacher_correct=True,
        prompt=f"PROMPT_{index}",
        completion="\nCC",
        parent_atom_count=index + 2,
        source_row_index=index,
    )


def _write_peft_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","task_type":"CAUSAL_LM",'
        '"base_model_name_or_path":"ChemLLM-7B-Chat"}',
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    return path


def test_prompt_completion_csv_loads_with_fixed_contract(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "train.csv",
        [_row("MUT_1", "CCO", "train", "TRAIN_SCAFFOLD")],
    )
    records = load_continued_sft_records(
        source, expected_split="train", expected_count=1
    )
    assert len(records) == 1
    assert records[0].prompt.endswith("FRAGMENT:")
    assert records[0].completion == "\nCC"
    assert records[0].molecule_id == "MUT_1"


def test_train_val_row_count_is_enforced_before_training(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "train.csv",
        [_row("MUT_1", "CCO", "train", "TRAIN_SCAFFOLD")],
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        load_continued_sft_records(
            source, expected_split="train", expected_count=1317
        )


def test_split_leakage_and_calibration_test_inputs_are_rejected(tmp_path: Path) -> None:
    calibration = _write(
        tmp_path / "calibration.csv",
        [_row("MUT_CAL", "CCO", "calibration", "CAL_SCAFFOLD")],
    )
    with pytest.raises(ValueError, match="Calibration/test inputs are forbidden"):
        load_continued_sft_records(
            calibration, expected_split="train", expected_count=1
        )

    train = load_continued_sft_records(
        _write(
            tmp_path / "train.csv",
            [_row("MUT_SHARED", "CCN", "train", "SHARED")],
        ),
        expected_split="train",
        expected_count=1,
    )
    val = load_continued_sft_records(
        _write(
            tmp_path / "val.csv",
            [_row("MUT_VAL", "CCC", "val", "SHARED")],
        ),
        expected_split="val",
        expected_count=1,
    )
    with pytest.raises(ValueError, match="scaffold_overlap"):
        validate_train_val_isolation(train, val)


def test_completion_tokens_participate_in_loss_and_prompt_is_masked() -> None:
    tokenized = tokenize_completion_only(
        FakeTokenizer(),
        _record(0),
        max_sequence_length=128,
    )
    assert tokenized.num_prompt_tokens > 0
    assert tokenized.num_supervised_tokens == tokenized.num_completion_tokens
    assert all(label == -100 for label in tokenized.labels[: tokenized.num_prompt_tokens])
    assert all(label != -100 for label in tokenized.labels[tokenized.num_prompt_tokens :])
    assert any(label != -100 for label in tokenized.labels)
    assert tokenized.input_ids[-1] == FakeTokenizer.eos_token_id


def test_completion_fully_truncated_is_a_hard_error() -> None:
    record = _record(0)
    with pytest.raises(ValueError, match="completion is fully truncated"):
        tokenize_completion_only(
            FakeTokenizer(),
            record,
            max_sequence_length=3,
        )


def test_smoke_sampling_is_deterministic_stratified_and_not_file_head() -> None:
    records = [_record(index) for index in range(20)]
    first = deterministic_smoke_sample(records, max_rows=6, seed=7)
    second = deterministic_smoke_sample(records, max_rows=6, seed=7)
    assert [row.molecule_id for row in first] == [row.molecule_id for row in second]
    assert [row.molecule_id for row in first] != [
        row.molecule_id for row in records[:6]
    ]
    assert len({row.stratum_key for row in first}) > 1


def test_checkpoint_manifest_records_initial_and_saved_peft_adapters(
    tmp_path: Path,
) -> None:
    base = _write_peft_checkpoint(tmp_path / "base" / "checkpoint-500")
    output = tmp_path / "output"
    first = _write_peft_checkpoint(output / "checkpoint-1")
    second = _write_peft_checkpoint(output / "checkpoint-3")
    manifest = build_checkpoint_manifest(
        output,
        initialization_checkpoint=base,
        best_checkpoint=second,
    )
    assert manifest["num_training_checkpoints"] == 2
    assert [row["step"] for row in manifest["checkpoints"]] == [1, 3]
    assert manifest["best_checkpoint"] == str(second.resolve())
    assert manifest["initialization_checkpoint"]["checkpoint"] == str(base.resolve())
    assert validate_peft_checkpoint(first)["peft_type"] == "LORA"


def test_parent_coverage_statistics_use_unique_ids_and_effective_batch() -> None:
    tracker = ParentCoverageTracker(["p1", "p2", "p3"])
    tracker.record(["p1", "p2"])
    tracker.record(["p2", "p3"])
    summary = tracker.summary(
        global_step=2,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        world_size=1,
        current_epoch=1.25,
    )
    assert summary["num_train_examples_seen"] == 4
    assert summary["num_unique_train_parents_seen"] == 3
    assert summary["unique_train_parent_coverage"] == 1.0
    assert summary["effective_batch_size"] == summary["samples_per_step"] == 16
    assert summary["epochs_equivalent"] == pytest.approx(4 / 3)


def test_missing_base_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="BASE checkpoint"):
        validate_peft_checkpoint(tmp_path / "missing" / "checkpoint-500")


def test_output_root_never_overwrites_existing_result(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "_RUN_COMPLETE.json").write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        ensure_new_output_root(output)
    fresh = ensure_new_output_root(tmp_path / "fresh")
    assert fresh.is_dir()


def test_continued_sft_config_uses_supported_yaml_and_expected_types() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "train"
        / "mutagenicity_continued_sft.yaml"
    )
    config = load_and_merge_config_files([config_path])

    assert config["data.preferred_root"].endswith("sft_ppo_data_v1")
    assert config["data.expected_train_rows"] == 1317
    assert config["data.expected_val_rows"] == 250
    assert config["data.forbidden_splits"] == "calibration,test"
    assert config["model.initialization"] == "continued_peft_adapter"
    assert config["training.max_sequence_length"] == 1024
    assert config["training.max_steps"] == 500
    assert config["training.per_device_train_batch_size"] == 4
    assert config["training.gradient_accumulation_steps"] == 4
    assert config["training.learning_rate"] == pytest.approx(0.0002)
    assert config["training.warmup_ratio"] == pytest.approx(0.03)
    assert config["training.bf16"] is True
    assert config["training.fp16"] is False
    assert config["training.report_to"] == "none"
    assert config["smoke.max_steps"] == 3
    assert config["full.output_root"].endswith("sft_continued_v1")

    assert _nested(config, "training", "max_steps") == 500
    assert _nested(config, "smoke", "max_steps") == 3
    assert _nested(config, "model", "base_model_path") == (
        "pretrained_models/ChemLLM-7B-Chat"
    )

    source_lines = config_path.read_text(encoding="utf-8").splitlines()
    config_lines = [
        line for line in source_lines if line.strip() and not line.startswith("#")
    ]
    assert all(line == line.lstrip() for line in config_lines)
    assert all(":" in line and line.partition(":")[2].strip() for line in config_lines)
    assert not any(line.startswith(("- ", "|", ">")) for line in config_lines)


def test_slurm_config_overrides_are_recognized_by_training_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser_options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    expected_overrides = {
        "--config",
        "--mode",
        "--data-root",
        "--train-csv",
        "--val-csv",
        "--base-model-path",
        "--base-checkpoint",
        "--tokenizer-path",
        "--output-root",
        "--max-train-rows",
        "--max-val-rows",
        "--max-steps",
        "--max-sequence-length",
        "--per-device-train-batch-size",
        "--per-device-eval-batch-size",
        "--gradient-accumulation-steps",
        "--learning-rate",
        "--logging-steps",
        "--save-steps",
        "--eval-steps",
        "--save-total-limit",
        "--generation-samples",
        "--seed",
    }

    for wrapper_name in (
        "train_mutagenicity_sft_smoke.sh",
        "train_mutagenicity_sft_full.sh",
    ):
        wrapper = (
            repo_root / "scripts" / "slurm" / wrapper_name
        ).read_text(encoding="utf-8")
        command = wrapper.split(
            "python scripts/train_mutagenicity_continued_sft.py", maxsplit=1
        )[1].split("\n\n", maxsplit=1)[0]
        wrapper_options = set(re.findall(r"--[a-z][a-z0-9-]*", command))
        assert wrapper_options == expected_overrides
        assert wrapper_options <= parser_options
