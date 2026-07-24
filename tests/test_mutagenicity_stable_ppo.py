from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.train_mutagenicity_ppo_stable import (
    DEFAULT_FULL_OUTPUT,
    DEFAULT_POLICY_ADAPTER,
    DEFAULT_SMOKE_OUTPUT,
    DEFAULT_TEACHER,
    LEGACY_SHARED_OUTPUT,
    build_parser,
    parse_args_with_config_precedence,
    print_and_validate_output_root_audit,
    resolve_output_root_audit,
)
from scripts.train_ppo import apply_config_overrides
from scripts.train_ppo_stable import (
    _evaluate_validation_set,
    _should_run_final_validation,
    _summarize_step_metrics,
    build_parser as build_aids_stable_parser,
    isolated_generation_rng,
    resolve_stable_config,
)
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer
from src.rewards.reward_wrapper_stable import (
    StableChemRLRewardWrapper,
    StableTeacherConfidenceGateConfig,
)
from src.train.mutagenicity_stable_ppo import (
    REQUIRED_CANDIDATE_FIELDS,
    MutagenicityCounterfactualTeacherScorer,
    MutagenicityPPORunObserver,
    audit_mutagenicity_ppo_models,
    build_parent_coverage_plan,
    deterministically_order_records,
    enrich_mutagenicity_candidate_row,
    load_mutagenicity_ppo_records,
    validate_candidate_pool_schema,
    validate_policy_adapter_checkpoint,
    validate_train_val_isolation,
)


def _write_prompt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    molecule_id: str,
    smiles: str,
    split: str,
    scaffold: str,
) -> dict[str, object]:
    return {
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
        "prompt": f"PARENT_SMILES: {smiles}\nORIGINAL_LABEL: 1",
    }


class _FakeParameter:
    def __init__(self, count: int, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _FakePeftModel:
    def __init__(
        self,
        *,
        adapters: tuple[str, ...] = ("default",),
        active: tuple[str, ...] = ("default",),
        trainable_lora: bool = True,
        trainable_base: bool = False,
    ) -> None:
        self.peft_config = {
            name: SimpleNamespace(peft_type="LORA") for name in adapters
        }
        self.active_adapters = list(active)
        self._parameters = [
            (
                "base_model.model.layers.0.weight",
                _FakeParameter(100, trainable_base),
            ),
            (
                "base_model.model.layers.0.lora_A.default.weight",
                _FakeParameter(10, trainable_lora),
            ),
        ]

    def named_parameters(self):
        return iter(self._parameters)


class _FakeValueModel:
    def __init__(self, *, base_trainable: bool = False) -> None:
        self._parameters = [
            ("pretrained_model.layer.weight", _FakeParameter(100, base_trainable)),
            ("v_head.summary.weight", _FakeParameter(8, True)),
        ]

    def named_parameters(self):
        return iter(self._parameters)


class _ValidationTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        prompts,
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
    ):
        del return_tensors, padding, truncation
        import torch

        input_ids = torch.tensor([[1, 2] for _ in prompts], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def batch_decode(self, payload, *, skip_special_tokens: bool):
        del skip_special_tokens
        return ["C" for _ in payload]


class _ValidationPolicy:
    def __init__(self, torch) -> None:
        self._torch = torch
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.generated_batches: list[list[list[int]]] = []
        self.training = True

    def parameters(self):
        return iter([self._parameter])

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, **kwargs):
        if "generator" in kwargs:
            raise AssertionError("generator must not be forwarded to model.generate")
        input_ids = kwargs["input_ids"]
        if bool(kwargs.get("do_sample")):
            response = self._torch.randint(
                3,
                10_000,
                (input_ids.shape[0], 4),
                device=input_ids.device,
            )
        else:
            response = self._torch.full(
                (input_ids.shape[0], 4),
                3,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        self.generated_batches.append(response.detach().cpu().tolist())
        return self._torch.cat([input_ids, response], dim=1)


class _ValidationRewardWrapper:
    def __init__(self, torch) -> None:
        self._torch = torch

    def compute_rewards_from_decoded(
        self,
        *,
        parent_smiles,
        generated_fragments,
        raw_outputs,
        labels,
        metas,
        device,
        step_index,
    ):
        del generated_fragments, raw_outputs, labels, metas, step_index
        rows = [
            {
                "parse_ok": True,
                "valid": True,
                "direct_substructure": True,
                "oracle_ok": True,
                "cf_flip": False,
                "cf_drop": 0.0,
                "total": 0.0,
                "atom_ratio": 0.25,
                "fragment": "C",
            }
            for _ in parent_smiles
        ]
        return self._torch.zeros(len(rows), device=device), rows


class _ValidationLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args) -> None:
        self.messages.append(message % args if args else message)


class _AtomRatioBaseRewarder:
    enable_size_window_reward = True
    size_window_low = 0.15
    size_window_high = 0.65

    def __init__(self, torch, row: dict[str, object]) -> None:
        self._torch = torch
        self._row = dict(row)

    def _compute_size_window_reward(self, *, atom_ratio):
        if atom_ratio is None:
            return 0.0, "unknown"
        if atom_ratio < self.size_window_low:
            return -0.4, "too_small"
        if atom_ratio > self.size_window_high:
            return -0.4, "too_large"
        return 0.4, "in_window"

    def compute_rewards_from_decoded(self, **kwargs):
        device = kwargs.get("device")
        reward = float(self._row["reward_total"])
        return self._torch.tensor([reward], device=device), [dict(self._row)]


def _validation_args(path: Path, *, seed: int, do_sample: bool):
    parser = build_aids_stable_parser()
    sample_flag = "--gen-do-sample" if do_sample else "--no-gen-do-sample"
    return parser.parse_args(
        [
            "--val-dataset-path",
            str(path),
            "--eval-num-samples",
            "2",
            "--batch-size",
            "2",
            "--seed",
            str(seed),
            "--gen-max-new-tokens",
            "4",
            "--gen-temperature",
            "1.0",
            "--gen-top-p",
            "1.0",
            sample_flag,
            "--default-parent-label",
            "1",
        ]
    )


def _run_fake_validation(
    path: Path,
    *,
    seed: int,
    do_sample: bool,
    step_index: int = 1,
):
    torch = pytest.importorskip("torch")
    args = _validation_args(path, seed=seed, do_sample=do_sample)
    policy = _ValidationPolicy(torch)
    logger = _ValidationLogger()
    summary = _evaluate_validation_set(
        deps={"torch": torch},
        args=args,
        stable_config=resolve_stable_config(args),
        policy_model=policy,
        tokenizer=_ValidationTokenizer(),
        reward_wrapper=_ValidationRewardWrapper(torch),
        step_index=step_index,
        logger=logger,
    )
    return policy.generated_batches, summary, logger.messages


def test_default_policy_is_mutagenicity_checkpoint_and_teacher() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert str(DEFAULT_POLICY_ADAPTER).endswith(
        "outputs/hpc/mutagenicity/final/sft_continued_v1_best"
    )
    assert "mutagenicity_rf_v1" in str(DEFAULT_TEACHER)
    assert args.default_parent_label == 1


def _write_ppo_config(path: Path, output_root: Path) -> None:
    path.write_text(
        f"paths:\n  output_root: {output_root}\n"
        "training:\n  batch_size: 16\n",
        encoding="utf-8",
    )


def test_explicit_cli_values_win_over_config_even_when_equal_to_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_root = tmp_path / "configured"
    _write_ppo_config(config_path, config_root)
    explicit_values = {
        "output_dir": str(DEFAULT_FULL_OUTPUT),
        "mode": "full",
        "max_parents": 1448,
        "rollout_batch_size": 64,
        "max_updates": 23,
        "eval_every_steps": 5,
        "train_csv": Path("train.csv"),
        "val_csv": Path("val.csv"),
        "teacher_path": "teacher.pkl",
        "base_model_path": Path("base"),
        "policy_adapter_checkpoint": Path("checkpoint-200"),
    }
    argv = [
        "--config",
        str(config_path),
        "--output-dir",
        explicit_values["output_dir"],
        "--mode",
        explicit_values["mode"],
        "--max-parents",
        str(explicit_values["max_parents"]),
        "--rollout-batch-size",
        str(explicit_values["rollout_batch_size"]),
        "--max-updates",
        str(explicit_values["max_updates"]),
        "--eval-every-steps",
        str(explicit_values["eval_every_steps"]),
        "--train-csv",
        str(explicit_values["train_csv"]),
        "--val-csv",
        str(explicit_values["val_csv"]),
        "--teacher-path",
        explicit_values["teacher_path"],
        "--base-model-path",
        str(explicit_values["base_model_path"]),
        "--policy-adapter-checkpoint",
        str(explicit_values["policy_adapter_checkpoint"]),
    ]

    _, args, audit = parse_args_with_config_precedence(argv)

    for destination, expected in explicit_values.items():
        assert getattr(args, destination) == expected
    assert audit["explicit_cli_preserved"] is True
    assert audit["config_candidates"]["output_dir"] == str(
        config_root / "rl_checkpoints"
    )


def test_config_output_applies_only_without_explicit_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_root = tmp_path / "configured"
    _write_ppo_config(config_path, config_root)

    _, args, audit = parse_args_with_config_precedence(
        ["--config", str(config_path)]
    )

    assert args.output_dir == str(config_root / "rl_checkpoints")
    assert "output_dir" not in audit["explicit_cli_destinations"]


@pytest.mark.parametrize(
    ("mode", "output_path"),
    [
        ("full", DEFAULT_FULL_OUTPUT),
        ("smoke", DEFAULT_SMOKE_OUTPUT),
    ],
)
def test_mutagenicity_output_root_preserves_explicit_task_path(
    mode: str,
    output_path: Path,
) -> None:
    _, args, priority_audit = parse_args_with_config_precedence(
        [
            "--config",
            "configs/hpc.yaml",
            "--mode",
            mode,
            "--output-dir",
            str(output_path),
        ]
    )
    resolved, output_audit = resolve_output_root_audit(args, priority_audit)

    assert resolved == output_path.resolve()
    assert output_audit["explicit_cli_preserved"] is True
    assert output_audit["output_root_audit_passed"] is True


def test_legacy_rl_output_cannot_replace_explicit_mutagenicity_output() -> None:
    _, args, priority_audit = parse_args_with_config_precedence(
        [
            "--mode",
            "full",
            "--output-dir",
            str(LEGACY_SHARED_OUTPUT),
        ]
    )
    _, output_audit = resolve_output_root_audit(args, priority_audit)

    with pytest.raises(AssertionError, match="legacy shared"):
        print_and_validate_output_root_audit(output_audit)


def test_checkpoint_200_is_required_and_checkpoint_500_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_200 = tmp_path / "checkpoint-200"
    checkpoint_200.mkdir()
    (checkpoint_200 / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "pretrained_models/ChemLLM-7B-Chat",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_200 / "adapter_model.bin").write_bytes(b"adapter")
    assert validate_policy_adapter_checkpoint(checkpoint_200)[
        "checkpoint_step_verified"
    ]

    checkpoint_500 = tmp_path / "checkpoint-500"
    checkpoint_500.mkdir()
    (checkpoint_500 / "adapter_config.json").write_text(
        (checkpoint_200 / "adapter_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (checkpoint_500 / "adapter_model.bin").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="checkpoint-200"):
        validate_policy_adapter_checkpoint(checkpoint_500)


def test_model_audit_accepts_one_policy_adapter_and_frozen_reference() -> None:
    audit = audit_mutagenicity_ppo_models(
        policy_model=_FakePeftModel(),
        reference_model=_FakePeftModel(trainable_lora=False),
        value_model=_FakeValueModel(),
        base_model_path="base",
        policy_adapter_checkpoint="checkpoint-200",
    )
    assert audit["policy_adapter_names"] == ["default"]
    assert audit["reference_trainable_params"] == 0
    assert audit["base_params_trainable"] == 0
    assert audit["value_head_trainable_params"] == 8


def test_model_audit_rejects_multiple_adapters() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        audit_mutagenicity_ppo_models(
            policy_model=_FakePeftModel(
                adapters=("default", "second"),
                active=("default",),
            ),
            reference_model=_FakePeftModel(trainable_lora=False),
            value_model=_FakeValueModel(),
            base_model_path="base",
            policy_adapter_checkpoint="checkpoint-200",
        )


def test_model_audit_rejects_trainable_reference_or_base() -> None:
    with pytest.raises(ValueError, match="Reference policy must be frozen"):
        audit_mutagenicity_ppo_models(
            policy_model=_FakePeftModel(),
            reference_model=_FakePeftModel(trainable_lora=True),
            value_model=_FakeValueModel(),
            base_model_path="base",
            policy_adapter_checkpoint="checkpoint-200",
        )
    with pytest.raises(ValueError, match="Base model parameters"):
        audit_mutagenicity_ppo_models(
            policy_model=_FakePeftModel(trainable_base=True),
            reference_model=_FakePeftModel(trainable_lora=False),
            value_model=_FakeValueModel(),
            base_model_path="base",
            policy_adapter_checkpoint="checkpoint-200",
        )


@pytest.mark.parametrize(
    ("pred_before", "pred_after", "expected"),
    [(1, 0, True), (0, 0, False), (1, 1, False)],
)
def test_mutagenicity_strict_flip_and_cf_drop(
    pred_before: int,
    pred_after: int,
    expected: bool,
) -> None:
    scorer = object.__new__(MutagenicityCounterfactualTeacherScorer)
    scorer.flip_bonus = 1.0
    with patch.object(
        CounterfactualTeacherScorer,
        "score_counterfactual",
        return_value={
            "teacher_result_ok": True,
            "p_before": 0.8,
            "p_after": 0.3,
            "pred_before": pred_before,
            "pred_after": pred_after,
            "cf_drop": 0.5,
            "cf_flip": True,
            "counterfactual_sem": 1.5,
            "teacher_sem": 1.5,
        },
    ):
        result = scorer.score_counterfactual("CCO", "C", label=1)
    assert result["cf_flip"] is expected
    assert result["cf_drop"] == pytest.approx(0.5)
    assert result["target_prob_gain"] == pytest.approx(0.5)
    assert result["prob_before_0"] == pytest.approx(0.2)
    assert result["prob_after_0"] == pytest.approx(0.7)


def test_train_val_contract_and_calibration_rejected(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    _write_prompt_csv(train_path, [_row("m1", "CCO", "train", "train_scaf")])
    _write_prompt_csv(val_path, [_row("m2", "CCN", "val", "val_scaf")])
    train = load_mutagenicity_ppo_records(
        train_path, expected_split="train", expected_count=1
    )
    val = load_mutagenicity_ppo_records(
        val_path, expected_split="val", expected_count=1
    )
    assert validate_train_val_isolation(train, val)["isolation_passed"]

    calibration_path = tmp_path / "calibration.csv"
    _write_prompt_csv(
        calibration_path,
        [_row("m3", "CCC", "train", "cal_scaf")],
    )
    with pytest.raises(ValueError, match="calibration/test"):
        load_mutagenicity_ppo_records(
            calibration_path, expected_split="train", expected_count=1
        )


def test_train_val_overlap_is_rejected(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    _write_prompt_csv(train_path, [_row("m1", "CCO", "train", "same_scaf")])
    _write_prompt_csv(val_path, [_row("m2", "CCN", "val", "same_scaf")])
    train = load_mutagenicity_ppo_records(
        train_path, expected_split="train", expected_count=1
    )
    val = load_mutagenicity_ppo_records(
        val_path, expected_split="val", expected_count=1
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_train_val_isolation(train, val)


def test_deterministic_smoke_order_and_parent_coverage_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.csv"
    rows = [
        _row(f"m{index}", "C" * (index + 1), "train", f"s{index}")
        for index in range(1, 7)
    ]
    _write_prompt_csv(path, rows)
    records = load_mutagenicity_ppo_records(
        path, expected_split="train", expected_count=6
    )
    first = deterministically_order_records(records, seed=7, limit=5)
    second = deterministically_order_records(records, seed=7, limit=5)
    assert [row.molecule_id for row in first] == [
        row.molecule_id for row in second
    ]
    plan = build_parent_coverage_plan(
        num_dataset_rows=5,
        rollout_batch_size=1,
        sampler_seed=7,
    )
    assert plan.samples_per_update == 1
    assert plan.updates_per_epoch == 5
    assert plan.max_updates == 5


def test_full_updates_per_epoch_uses_real_rollout_batch() -> None:
    plan = build_parent_coverage_plan(
        num_dataset_rows=1448,
        rollout_batch_size=64,
        sampler_seed=7,
    )
    assert plan.samples_per_update == 64
    assert plan.updates_per_epoch == math.ceil(1448 / 64) == 23

    explicit = build_parent_coverage_plan(
        num_dataset_rows=1448,
        rollout_batch_size=64,
        sampler_seed=7,
        max_updates=29,
    )
    assert explicit.updates_per_epoch == 23
    assert explicit.max_updates == 29


def test_candidate_enrichment_contains_directional_teacher_fields() -> None:
    row = enrich_mutagenicity_candidate_row(
        {
            "parent_smiles": "CCO",
            "p_before": 0.8,
            "p_after": 0.2,
            "pred_before": 1,
            "pred_after": 0,
            "oracle_ok": True,
            "raw_parse_ok": True,
            "direct_substructure": True,
            "parent_without_fragment_smiles": "CO",
            "total": 2.0,
            "breakdown": {"cf": 1.0},
        },
        molecule_id="m1",
        prompt="prompt",
        generated_text="C",
        generated_fragment="C",
        global_step=1,
    )
    assert row["cf_flip"] is True
    assert row["cf_drop"] == pytest.approx(0.6)
    assert row["target_prob_gain"] == pytest.approx(0.6)
    assert row["residual_smiles"] == "CO"
    assert set(REQUIRED_CANDIDATE_FIELDS) <= set(row)
    validate_candidate_pool_schema([row])


def test_smoke_observer_tracks_five_unique_parents_without_replacement(
    tmp_path: Path,
) -> None:
    ids = [f"m{index}" for index in range(5)]
    plan = build_parent_coverage_plan(
        num_dataset_rows=5,
        rollout_batch_size=1,
        sampler_seed=7,
    )
    observer = MutagenicityPPORunObserver(
        output_root=tmp_path,
        dataset_parent_ids=ids,
        coverage_plan=plan,
        resolved_config={
            "policy_adapter_checkpoint": "checkpoint-200",
            "base_model_path": "base",
            "teacher_path": "teacher.pkl",
        },
        dataset_manifest={
            "num_train_rows_selected": 5,
            "num_val_rows_selected": 2,
        },
        require_full_coverage=True,
    )
    for step, molecule_id in enumerate(ids, start=1):
        observer.on_update(
            step_index=step,
            batch_ids=[molecule_id],
            reward_logs=[],
            metrics={"reward_mean": 0.0},
        )
    summary = observer.coverage_summary(global_step=5)
    assert summary["num_samples_processed"] == 5
    assert summary["num_unique_parents_seen"] == 5
    assert summary["unique_parent_coverage"] == pytest.approx(1.0)
    assert summary["sampling_with_replacement"] is False


def test_aids_stable_entry_remains_available() -> None:
    args = build_aids_stable_parser().parse_args([])
    assert args.ppo_loop in {"decoded_chem", "trl_experimental"}
    assert hasattr(args, "teacher_path")


def test_aids_stable_config_merge_preserves_explicit_cli(
    tmp_path: Path,
) -> None:
    parser = build_aids_stable_parser()
    explicit_output = parser.get_default("output_dir")
    config_path = tmp_path / "config.yaml"
    _write_ppo_config(config_path, tmp_path / "configured")
    argv = [
        "--config",
        str(config_path),
        "--output-dir",
        explicit_output,
    ]
    args = parser.parse_args(argv)

    merged = apply_config_overrides(args, parser, argv=argv)

    assert merged.output_dir == explicit_output


def test_validation_generation_is_reproducible_without_generator_kwarg(
    tmp_path: Path,
) -> None:
    val_path = tmp_path / "val.csv"
    _write_prompt_csv(
        val_path,
        [
            _row("m1", "CCO", "val", "scaf1"),
            _row("m2", "CCN", "val", "scaf2"),
        ],
    )

    first, first_summary, messages = _run_fake_validation(
        val_path,
        seed=17,
        do_sample=True,
    )
    second, second_summary, _ = _run_fake_validation(
        val_path,
        seed=17,
        do_sample=True,
    )
    different, _, _ = _run_fake_validation(
        val_path,
        seed=18,
        do_sample=True,
    )

    assert first == second
    assert first != different
    assert first_summary == second_summary
    assert any(
        "rng_isolated=true" in message
        and "generator_kwarg_forwarded=false" in message
        for message in messages
    )


def test_validation_generation_restores_global_cpu_rng_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    val_path = tmp_path / "val.csv"
    _write_prompt_csv(val_path, [_row("m1", "CCO", "val", "scaf1")])
    torch.manual_seed(12345)
    state_before = torch.random.get_rng_state().clone()

    _run_fake_validation(val_path, seed=17, do_sample=True)

    assert torch.equal(state_before, torch.random.get_rng_state())


def test_isolated_generation_rng_restores_cuda_rng_state_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.cuda.manual_seed_all(12345)
    states_before = [state.clone() for state in torch.cuda.get_rng_state_all()]

    with isolated_generation_rng(
        torch=torch,
        seed=17,
        model_device=torch.device("cuda", torch.cuda.current_device()),
    ):
        torch.rand(8, device="cuda")

    states_after = torch.cuda.get_rng_state_all()
    assert len(states_before) == len(states_after)
    assert all(
        torch.equal(before, after)
        for before, after in zip(states_before, states_after, strict=True)
    )


def test_greedy_validation_does_not_forward_generator(tmp_path: Path) -> None:
    val_path = tmp_path / "val.csv"
    _write_prompt_csv(val_path, [_row("m1", "CCO", "val", "scaf1")])

    generated, summary, _ = _run_fake_validation(
        val_path,
        seed=17,
        do_sample=False,
    )

    assert generated == [[[3, 3, 3, 3]]]
    assert summary is not None


def test_direct_substructure_uses_final_fragment_atom_ratio() -> None:
    torch = pytest.importorskip("torch")
    row = {
        "parent_smiles": "CCCCCC",
        "raw_fragment": "CCC",
        "core_fragment": "CCC",
        "direct_substructure": True,
        "used_projected_subgraph_for_reward": False,
        "size_window_reward": 0.4,
        "reward_total": 2.0,
        "total": 2.0,
        "reward_components": {"size_window_r": 0.4},
    }
    wrapper = StableChemRLRewardWrapper(
        base_rewarder=_AtomRatioBaseRewarder(torch, row),
        teacher_conf_gate=StableTeacherConfidenceGateConfig(enabled=False),
    )

    reward_tensor, logs = wrapper.compute_rewards_from_decoded(
        parent_smiles=["CCCCCC"],
        generated_fragments=["CCC"],
        device=torch.device("cpu"),
    )

    assert reward_tensor.item() == pytest.approx(2.0)
    assert logs[0]["final_fragment"] == "CCC"
    assert logs[0]["parent_heavy_atoms"] == 6
    assert logs[0]["final_fragment_heavy_atoms"] == 3
    assert logs[0]["atom_ratio"] == pytest.approx(0.5)
    assert logs[0]["atom_ratio_source"] == "final_fragment"
    assert logs[0]["reward_components"]["size_window_r"] == pytest.approx(0.4)


def test_projection_ratio_drives_reward_candidate_and_update_metrics() -> None:
    torch = pytest.importorskip("torch")
    parent = "CCCCCCCCCC"
    raw_fragment = "CCCCCCCCCCCCCCC"
    projected_fragment = "CCCC"
    row = {
        "parent_smiles": parent,
        "raw_fragment": raw_fragment,
        "core_fragment": raw_fragment,
        "projected_fragment": projected_fragment,
        "nearest_parent_subgraph_smiles": projected_fragment,
        "projection_method": "nearest_parent_subgraph",
        "projection_success": True,
        "used_projected_subgraph_for_reward": True,
        "direct_substructure": False,
        "size_window_reward": -0.4,
        "reward_total": 1.0,
        "total": 1.0,
        "reward_components": {"size_window_r": -0.4},
    }
    wrapper = StableChemRLRewardWrapper(
        base_rewarder=_AtomRatioBaseRewarder(torch, row),
        teacher_conf_gate=StableTeacherConfidenceGateConfig(enabled=False),
    )

    reward_tensor, logs = wrapper.compute_rewards_from_decoded(
        parent_smiles=[parent],
        generated_fragments=[raw_fragment],
        device=torch.device("cpu"),
    )
    log = logs[0]
    candidate = enrich_mutagenicity_candidate_row(
        log,
        molecule_id="m1",
        prompt="prompt",
        generated_text=raw_fragment,
        generated_fragment=raw_fragment,
        global_step=1,
        parent_smiles=parent,
    )
    metrics = _summarize_step_metrics([candidate])

    assert log["raw_fragment_heavy_atoms"] == 15
    assert log["raw_atom_ratio"] == pytest.approx(1.5)
    assert log["final_fragment"] == projected_fragment
    assert log["final_fragment_heavy_atoms"] == 4
    assert log["atom_ratio"] == pytest.approx(0.4)
    assert log["atom_ratio"] <= 1.0
    assert log["atom_ratio_source"] == "final_fragment"
    assert log["size_window_reward"] == pytest.approx(0.4)
    assert log["reward_components"]["size_window_r"] == pytest.approx(0.4)
    assert reward_tensor.item() == pytest.approx(1.8)
    assert candidate["atom_ratio"] == pytest.approx(0.4)
    assert candidate["raw_atom_ratio"] == pytest.approx(1.5)
    assert candidate["atom_ratio_source"] == "final_fragment"
    assert candidate["reward_components"]["size_window_r"] == pytest.approx(0.4)
    assert metrics["atom_ratio_mean"] == pytest.approx(0.4)


def test_final_substructure_atom_ratio_above_one_fails_audit() -> None:
    torch = pytest.importorskip("torch")
    row = {
        "parent_smiles": "C",
        "raw_fragment": "CC",
        "core_fragment": "CC",
        "direct_substructure": True,
        "size_window_reward": -0.4,
        "reward_total": 1.0,
        "total": 1.0,
    }
    wrapper = StableChemRLRewardWrapper(
        base_rewarder=_AtomRatioBaseRewarder(torch, row),
        teacher_conf_gate=StableTeacherConfidenceGateConfig(enabled=False),
    )

    with patch(
        "src.rewards.reward_wrapper_stable.is_parent_substructure",
        return_value=True,
    ), pytest.raises(RuntimeError, match="STABLE_PPO_ATOM_RATIO_AUDIT_FAILED"):
        wrapper.compute_rewards_from_decoded(
            parent_smiles=["C"],
            generated_fragments=["CC"],
            device=torch.device("cpu"),
        )


def test_full_validation_runs_at_last_non_interval_step(tmp_path: Path) -> None:
    args = _validation_args(tmp_path / "val.csv", seed=7, do_sample=False)
    args.eval_every_steps = 100
    config = resolve_stable_config(args)

    assert _should_run_final_validation(
        stable_config=config,
        completed_steps=1448,
        last_validation_step=1400,
    )
    assert not _should_run_final_validation(
        stable_config=config,
        completed_steps=1400,
        last_validation_step=1400,
    )


def test_full_wrapper_defaults_to_eval_every_100_and_smoke_remains_every_step() -> None:
    root = Path(__file__).resolve().parents[1]
    full_script = (
        root / "scripts/slurm/train_mutagenicity_ppo_stable_full.sh"
    ).read_text(encoding="utf-8")
    smoke_script = (
        root / "scripts/slurm/train_mutagenicity_ppo_stable_smoke.sh"
    ).read_text(encoding="utf-8")

    assert 'EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"' in full_script
    assert 'MAX_UPDATES="${MAX_UPDATES:-$UPDATES_PER_EPOCH}"' in full_script
    assert "--eval-every-steps 1" in smoke_script
