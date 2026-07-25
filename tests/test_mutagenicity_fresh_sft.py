from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.data.mutagenicity_sft_v2 import (
    ParentCandidateSet,
    build_strict_multitarget_rows,
    is_positive_fallback_candidate,
    is_teacher_strict_candidate,
    select_strict_first_candidate,
)
from src.data.sft_v3_builder import SFTV3ReferenceCandidate
from src.data.mutagenicity_sft_ppo import MutagenicityParent
from src.train.mutagenicity_fresh_sft import (
    FreshLoRAConfig,
    assert_no_adapter_files_read,
    assert_pure_base_model,
    audit_fresh_lora_model,
    fresh_checkpoint_manifest,
    initialize_fresh_lora,
    resolve_variant_csvs,
    tokenizer_reuse_audit,
)


class FakeParameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class FakeBase:
    def __init__(self, *, adapted: bool = False) -> None:
        if adapted:
            self.peft_config = {"old": object()}
        self._parameters = [("model.weight", FakeParameter(100, requires_grad=False))]

    def named_parameters(self):
        return list(self._parameters)


class FakeFreshModel:
    def __init__(self) -> None:
        self.peft_config = {"default": {"peft_type": "LORA"}}
        self.active_adapters = ["default"]
        self._parameters = [
            ("base_model.model.weight", FakeParameter(1000, requires_grad=False)),
            ("base_model.model.layer.lora_A.default.weight", FakeParameter(20, requires_grad=True)),
            ("base_model.model.layer.lora_B.default.weight", FakeParameter(20, requires_grad=True)),
        ]

    def named_parameters(self):
        return list(self._parameters)


def _candidate(
    *,
    strategy: str,
    pred_after: int,
    cf_drop: float,
    atom_ratio: float,
    fragment: str,
) -> SFTV3ReferenceCandidate:
    return SFTV3ReferenceCandidate(
        core_fragment=fragment,
        raw_fragment=fragment,
        candidate_strategy=strategy,
        atom_count=2,
        atom_ratio=atom_ratio,
        residual_smiles="C",
        residual_nonempty=True,
        is_full_parent=False,
        oracle_ok=True,
        cf_flip=pred_after == 0,
        cf_drop=cf_drop,
        pred_before=1,
        pred_after=pred_after,
        p_before=0.9,
        p_after=0.9 - cf_drop,
    )


def test_fresh_lora_wraps_pure_base_once() -> None:
    calls = []
    fresh = FakeFreshModel()

    def wrapper(base, config):
        calls.append((base, config))
        return fresh

    model, loading = initialize_fresh_lora(
        FakeBase(),
        lora_config={"r": 8},
        get_peft_model_fn=wrapper,
    )
    assert model is fresh
    assert len(calls) == 1
    assert loading["source_adapter_checkpoint"] is None
    assert loading["aids_adapter_weights_loaded"] is False


def test_fresh_lora_rejects_preexisting_adapter() -> None:
    with pytest.raises(ValueError, match="pure ChemLLM base"):
        assert_pure_base_model(FakeBase(adapted=True))


def test_fresh_initialization_audit_requires_single_frozen_base_adapter(tmp_path: Path) -> None:
    settings = FreshLoRAConfig()
    payload = audit_fresh_lora_model(
        FakeFreshModel(),
        base_model_path=tmp_path,
        loading_audit={
            "source_adapter_checkpoint": None,
            "aids_adapter_weights_loaded": False,
        },
        lora_settings=settings,
    )
    assert payload["initialization_audit_passed"] is True
    assert payload["adapter_names"] == ["default"]
    assert payload["base_parameter_trainable_count"] == 0
    assert payload["adapter_trainable_parameter_count"] == 40


def test_tokenizer_fallback_is_not_adapter_weight_reuse(tmp_path: Path) -> None:
    @dataclass
    class Tokenizer:
        bos_token_id: int = 1
        eos_token_id: int = 2
        pad_token_id: int = 2
        unk_token_id: int = 0

        def __len__(self) -> int:
            return 100

    audit = tokenizer_reuse_audit(
        tokenizer=Tokenizer(),
        base_model_path=tmp_path / "base",
        tokenizer_path=tmp_path / "old_checkpoint",
    )
    assert audit["tokenizer_reused"] is True
    assert audit["adapter_weights_reused"] is False
    with pytest.raises(ValueError, match="must not read adapter files"):
        assert_no_adapter_files_read([tmp_path / "adapter_model.safetensors"])


def test_fresh_checkpoint_manifest_covers_final_and_periodic_adapters(
    tmp_path: Path,
) -> None:
    for root in (tmp_path, tmp_path / "checkpoint-50"):
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"fresh")
    manifest = fresh_checkpoint_manifest(
        tmp_path,
        best_token_loss_checkpoint=tmp_path / "checkpoint-50",
    )
    assert manifest["source_adapter_checkpoint"] is None
    assert manifest["final_adapter"]["checkpoint"] == str(tmp_path.resolve())
    assert len(manifest["checkpoints"]) == 1


def test_strict_candidate_hard_priority_over_non_strict_bond_k2() -> None:
    non_strict = _candidate(
        strategy="bond_k2",
        pred_after=1,
        cf_drop=0.80,
        atom_ratio=0.10,
        fragment="CC",
    )
    strict = _candidate(
        strategy="ring",
        pred_after=0,
        cf_drop=0.20,
        atom_ratio=0.40,
        fragment="C1CC1",
    )
    selected = select_strict_first_candidate(
        [non_strict, strict],
        completion_frequency={"CC": 1, "C1CC1": 10},
        allow_soft_fallback=True,
    )
    assert selected.selected is strict
    assert selected.target_quality == "strict"
    assert is_teacher_strict_candidate(strict)


def test_fallback_only_when_no_strict_candidate() -> None:
    weak = _candidate(
        strategy="bond_k2",
        pred_after=1,
        cf_drop=0.40,
        atom_ratio=0.30,
        fragment="CO",
    )
    selected = select_strict_first_candidate(
        [weak],
        completion_frequency={"CO": 1},
        allow_soft_fallback=True,
    )
    assert selected.selected is weak
    assert selected.target_quality == "soft_fallback"
    assert is_positive_fallback_candidate(weak)
    rejected = select_strict_first_candidate(
        [weak],
        completion_frequency={"CO": 1},
        allow_soft_fallback=False,
    )
    assert rejected.selected is None


def test_strict_multitarget_deduplicates_within_parent_and_keeps_split() -> None:
    parent = MutagenicityParent(
        molecule_id="MUT_TEST",
        source_row_index=1,
        source_smiles="CCCC",
        parent_smiles="CCCC",
        label=1,
        semantic_label="mutagenic",
        split="train",
        scaffold_smiles="",
        teacher_pred=1,
        teacher_prob_0=0.1,
        teacher_prob_1=0.9,
        teacher_correct=True,
        parent_atom_count=4,
    )
    duplicate_a = _candidate(
        strategy="bond_k1",
        pred_after=0,
        cf_drop=0.4,
        atom_ratio=0.5,
        fragment="CC",
    )
    duplicate_b = _candidate(
        strategy="bond_k2",
        pred_after=0,
        cf_drop=0.5,
        atom_ratio=0.5,
        fragment="C-C",
    )
    distinct = _candidate(
        strategy="atom_k1",
        pred_after=0,
        cf_drop=0.3,
        atom_ratio=0.25,
        fragment="C",
    )
    rows = build_strict_multitarget_rows(
        [
            ParentCandidateSet(
                parent=parent,
                candidates=(duplicate_a, duplicate_b, distinct),
                proposal_count=3,
                drop_reason=None,
            )
        ],
        frequencies={"CC": 2, "C": 1},
        max_targets_per_parent=3,
        max_completion_frequency=None,
    )
    assert len(rows) == 2
    assert {row["split"] for row in rows} == {"train"}
    assert {row["molecule_id"] for row in rows} == {"MUT_TEST"}
    assert len({row["completion"] for row in rows}) == 2


@pytest.mark.parametrize(
    ("variant", "train_name"),
    [
        ("current_v1", "mutagenicity_sft_train.csv"),
        ("strict_v2", "mutagenicity_sft_train_strict_v2.csv"),
        ("fallback_v2", "mutagenicity_sft_train_fallback_v2.csv"),
        (
            "strict_multitarget_v2",
            "mutagenicity_sft_train_strict_multitarget_v2.csv",
        ),
    ],
)
def test_dataset_variant_files_are_explicit(
    tmp_path: Path, variant: str, train_name: str
) -> None:
    train, val = resolve_variant_csvs(tmp_path, variant)
    assert train.name == train_name
    assert val.parent == tmp_path.resolve()


def _wrapper_text(name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "slurm" / name).read_text(encoding="utf-8")


def _resource_directives(text: str) -> set[str]:
    return {
        line
        for line in text.splitlines()
        if line.startswith("#SBATCH ") and "--job-name=" not in line
    }


def test_strict_v2_smoke_wrapper_has_isolated_dataset_and_output() -> None:
    text = _wrapper_text("train_mutagenicity_sft_fresh_strict_v2_smoke.sh")

    assert "current_v1" not in text
    assert "sft_ppo_data_v1" not in text
    assert "sft_ppo_data_v2" in text
    assert "DATASET_VARIANT=strict_v2" in text
    assert "--dataset-variant strict_v2" in text
    assert "sft_fresh_strict_v2_smoke" in text
    assert "[MUTAGENICITY_FRESH_SFT_STRICT_V2_SMOKE_OK]" in text
    assert 'test -s "$OUTPUT_ROOT/_RUN_COMPLETE.json"' in text


def test_strict_v2_smoke_wrapper_reuses_current_resource_contract() -> None:
    current = _wrapper_text("train_mutagenicity_sft_fresh_smoke.sh")
    strict_v2 = _wrapper_text(
        "train_mutagenicity_sft_fresh_strict_v2_smoke.sh"
    )

    assert _resource_directives(strict_v2) == _resource_directives(current)
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"' in strict_v2


def test_strict_v2_tokenizer_fallback_is_never_loaded_as_adapter() -> None:
    text = _wrapper_text("train_mutagenicity_sft_fresh_strict_v2_smoke.sh")
    train_block, audit_block = text.split(
        "python scripts/audit_mutagenicity_sft_fresh.py", maxsplit=1
    )

    assert (
        '--tokenizer-fallback-path "$TOKENIZER_FALLBACK_PATH"' in train_block
    )
    assert "--adapter-checkpoint" not in train_block
    assert "--source-adapter-checkpoint" not in train_block
    assert (
        '--forbidden-adapter-checkpoint "$TOKENIZER_FALLBACK_PATH"'
        in audit_block
    )


def test_current_v1_smoke_wrapper_contract_remains_separate() -> None:
    text = _wrapper_text("train_mutagenicity_sft_fresh_smoke.sh")

    assert "--dataset-variant current_v1" in text
    assert "sft_ppo_data_v1" in text
    assert "sft_fresh_v1_smoke" in text
    assert "strict_v2" not in text
