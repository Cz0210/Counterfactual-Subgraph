from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ablations.llm import (
    AVAILABLE,
    BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
    ArtifactPin,
    AttemptRegime,
    BRICSFixedGenerator,
    ChemLLMGeneratorAdapter,
    GeneratorAssets,
    LLMAblationContractError,
    LLMProposerVariant,
    MatchedAttemptBudget,
    ParentInput,
    ProposalRequest,
    ProposalResult,
    RuntimeGeneratorIdentity,
    artifact_sha256,
    build_common_downstream_plan,
    build_proposal_record,
    build_train_only_brics_vocabulary,
    proposal_output_template,
    run_manifest_template,
    summarize_novelty,
    training_molecules_from_mappings,
    validate_attempt_matched_schedules,
    validate_proposal_record,
)
from src.models.interfaces import GenerationResult
from src.ablations.llm.contracts import canonical_json_sha256
from scripts.ablations.llm.build_llm_ablation_plan import build_framework_payloads


SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _pin(path: Path, role: str) -> ArtifactPin:
    return ArtifactPin(path=str(path), sha256=artifact_sha256(path), role=role)


def _request(**overrides) -> ProposalRequest:
    payload = {
        "parent_id": "bace-1",
        "parent_smiles": "CCOC(=O)NCC",
        "source_label": 1,
        "regime": "base",
        "regime_attempt_index": 0,
        "attempt_index": 0,
        "attempt_seed": 7,
        "attempt_budget_sha256": "2" * 64,
        "max_new_tokens": 96,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    payload.update(overrides)
    if "slot_id" not in payload:
        payload["slot_id"] = canonical_json_sha256(
            {
                key: payload[key]
                for key in (
                    "parent_id",
                    "regime",
                    "regime_attempt_index",
                    "attempt_index",
                    "attempt_seed",
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                )
            }
        )
    return ProposalRequest(**payload)


def _vocabulary():
    pytest.importorskip("rdkit")
    return build_train_only_brics_vocabulary(
        training_molecules_from_mappings(
            [
                {
                    "molecule_id": "m1",
                    "smiles": "CCOC(=O)NCC",
                    "split": "train",
                    "label": "1",
                },
                {
                    "molecule_id": "m2",
                    "smiles": "CCN(CC)CCOC(=O)c1ccccc1",
                    "split": "train",
                    "label": "0",
                },
            ]
        )
    )


def test_generator_asset_topology_and_missing_paths_fail_closed(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer.model"
    sft = tmp_path / "sft"
    ppo = tmp_path / "ppo"
    for path in (base, sft, ppo):
        path.mkdir()
    tokenizer.write_text("tokenizer", encoding="utf-8")

    pretrained = GeneratorAssets(
        variant=LLMProposerVariant.CHEMLLM_PRETRAINED,
        base_model=_pin(base, "base"),
        tokenizer=_pin(tokenizer, "tokenizer"),
    )
    assert pretrained.load_adapter is None

    sft_only = GeneratorAssets(
        variant=LLMProposerVariant.CHEMLLM_SFT,
        base_model=_pin(base, "base"),
        tokenizer=_pin(tokenizer, "tokenizer"),
        sft_adapter=_pin(sft, "sft"),
    )
    assert sft_only.load_adapter is sft_only.sft_adapter
    with pytest.raises(LLMAblationContractError):
        replace(sft_only, ppo_adapter=_pin(ppo, "ppo"))

    sft_ppo = GeneratorAssets(
        variant=LLMProposerVariant.CHEMLLM_SFT_PPO,
        base_model=_pin(base, "base"),
        tokenizer=_pin(tokenizer, "tokenizer"),
        sft_adapter=_pin(sft, "sft"),
        ppo_adapter=_pin(ppo, "ppo"),
    )
    assert sft_ppo.load_adapter is sft_ppo.ppo_adapter
    with pytest.raises(LLMAblationContractError):
        GeneratorAssets(
            variant=LLMProposerVariant.CHEMLLM_SFT_PPO,
            base_model=_pin(base, "base"),
            tokenizer=_pin(tokenizer, "tokenizer"),
            sft_adapter=_pin(sft, "sft"),
        )
    with pytest.raises(LLMAblationContractError, match="does not exist"):
        ArtifactPin(path=str(tmp_path / "missing"), sha256=SHA, role="missing")


def test_artifact_pin_recomputes_file_and_directory_hashes(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"exact bytes")
    ArtifactPin(
        path=str(file_path),
        sha256=artifact_sha256(file_path),
        role="file",
    )
    with pytest.raises(LLMAblationContractError, match="sha256 mismatch"):
        ArtifactPin(path=str(file_path), sha256=SHA, role="file")

    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text("{}\n", encoding="utf-8")
    nested = directory / "weights"
    nested.mkdir()
    (nested / "part.bin").write_bytes(b"weights")
    first_sha = artifact_sha256(directory)
    ArtifactPin(path=str(directory), sha256=first_sha, role="directory")
    (nested / "part.bin").write_bytes(b"changed")
    assert artifact_sha256(directory) != first_sha
    with pytest.raises(LLMAblationContractError, match="sha256 mismatch"):
        ArtifactPin(path=str(directory), sha256=first_sha, role="directory")


def test_current_sft_variants_are_explicitly_blocked_without_matched_sft() -> None:
    config = json.loads(
        (
            REPO_ROOT
            / "configs/ablations/llm/llm_proposer_ablation_v1.template.json"
        ).read_text(encoding="utf-8")
    )
    for variant in ("CHEMLLM_SFT", "CHEMLLM_SFT_PPO"):
        entry = config["generator_assets"][variant]
        assert entry["availability"] == BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT
        assets = GeneratorAssets.from_mapping(variant, entry)
        assert assets.availability == BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT
        with pytest.raises(LLMAblationContractError, match="fail-closed"):
            ChemLLMGeneratorAdapter(generator=_LegacyFakeFragmentGenerator(), assets=assets)


class _LegacyFakeFragmentGenerator:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return GenerationResult(
            fragment_smiles="CC",
            raw_text="CC",
            finish_reason="eos",
            metadata={"fake": True},
        )


class _BatchFakeFragmentGenerator(_LegacyFakeFragmentGenerator):
    def __init__(self, assets: GeneratorAssets) -> None:
        super().__init__()
        self.regime_calls = []
        self._identity = RuntimeGeneratorIdentity(
            assets=assets,
            generation_api="single_call_num_return_sequences_v1",
            explicit_seed_supported=True,
            num_return_sequences_supported=True,
        )

    @property
    def llm_ablation_runtime_identity(self):
        return self._identity

    def generate_ablation_regime(self, **kwargs):
        self.regime_calls.append(kwargs)
        return tuple(
            GenerationResult(
                fragment_smiles=f"C{'C' * index}",
                raw_text=f"C{'C' * index}",
                finish_reason="eos",
                metadata={"sequence_index": index},
            )
            for index in range(4)
        )


def test_unified_generator_adapts_existing_chemlm_interface(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer.model"
    base.mkdir()
    tokenizer.write_text("tokenizer", encoding="utf-8")
    assets = GeneratorAssets(
        variant=LLMProposerVariant.CHEMLLM_PRETRAINED,
        base_model=_pin(base, "base"),
        tokenizer=_pin(tokenizer, "tokenizer"),
        availability=AVAILABLE,
    )
    with pytest.raises(LLMAblationContractError, match="CONFIG_ONLY"):
        ChemLLMGeneratorAdapter(generator=_LegacyFakeFragmentGenerator(), assets=assets)

    other_base = tmp_path / "other_base"
    other_base.mkdir()
    (other_base / "config.json").write_text("{}\n", encoding="utf-8")
    mismatched_runtime_assets = GeneratorAssets(
        variant=LLMProposerVariant.CHEMLLM_PRETRAINED,
        base_model=_pin(other_base, "runtime_base"),
        tokenizer=_pin(tokenizer, "runtime_tokenizer"),
    )
    with pytest.raises(LLMAblationContractError, match="runtime base_model identity mismatch"):
        ChemLLMGeneratorAdapter(
            generator=_BatchFakeFragmentGenerator(mismatched_runtime_assets),
            assets=assets,
        )

    fake = _BatchFakeFragmentGenerator(assets)
    adapter = ChemLLMGeneratorAdapter(generator=fake, assets=assets)
    with pytest.raises(LLMAblationContractError, match="generate_regime"):
        adapter.generate(_request())
    requests = tuple(
        _request(
            regime_attempt_index=index,
            attempt_index=index,
        )
        for index in range(4)
    )
    results = adapter.generate_regime(requests)
    assert len(fake.regime_calls) == 1
    assert fake.regime_calls[0]["num_return_sequences"] == 4
    assert fake.regime_calls[0]["seed"] == 7
    assert len(results) == 4
    assert all(result.variant is LLMProposerVariant.CHEMLLM_PRETRAINED for result in results)
    assert results[0].metadata["adapter_load_role"] is None


def test_brics_vocabulary_is_train_only_deterministic_and_oracle_blind() -> None:
    pytest.importorskip("rdkit")
    rows = [
        {
            "molecule_id": "m2",
            "smiles": "CCN(CC)CCOC(=O)c1ccccc1",
            "split": "train",
            "label": "0",
            "oracle_score": 0.99,
        },
        {
            "molecule_id": "m1",
            "smiles": "CCOC(=O)NCC",
            "split": "train",
            "label": "1",
            "oracle_score": 0.01,
        },
    ]
    first = build_train_only_brics_vocabulary(training_molecules_from_mappings(rows))
    swapped = [dict(row, oracle_score=1.0 - row["oracle_score"]) for row in reversed(rows)]
    second = build_train_only_brics_vocabulary(training_molecules_from_mappings(swapped))
    assert first.to_dict() == second.to_dict()
    assert len(first.fragments) == len(set(first.fragments))
    assert [record.vocabulary_rank for record in first.records] == list(
        range(1, len(first.records) + 1)
    )
    assert list(first.records) == sorted(
        first.records,
        key=lambda record: (-record.train_frequency, record.fragment_smiles),
    )
    assert first.ranking_policy == "train_frequency_only_no_oracle"
    assert first.oracle_fields_read == ()

    bad = [dict(rows[0], split="validation")]
    with pytest.raises(LLMAblationContractError, match="train-only"):
        training_molecules_from_mappings(bad)


def test_brics_generator_is_deterministic_and_never_uses_oracle() -> None:
    vocabulary = _vocabulary()
    generator = BRICSFixedGenerator(vocabulary)
    first = generator.generate(_request())
    second = generator.generate(_request())
    assert first == second
    assert first.variant is LLMProposerVariant.BRICS_FIXED
    assert first.metadata["oracle_used"] is False
    assert first.metadata["selection_policy"] == "parent_match_then_train_frequency"
    parent = pytest.importorskip("rdkit").Chem.MolFromSmiles(_request().parent_smiles)
    fragment = pytest.importorskip("rdkit").Chem.MolFromSmiles(first.fragment_smiles)
    assert parent.HasSubstructMatch(fragment)


def test_brics_generator_records_shortfall_without_duplicate_candidates() -> None:
    vocabulary = _vocabulary()
    generator = BRICSFixedGenerator(vocabulary)
    results = [
        generator.generate(_request(attempt_index=index))
        for index in range(len(vocabulary.records) + 2)
    ]
    emitted = [result.fragment_smiles for result in results if result.fragment_smiles]
    assert len(emitted) == len(set(emitted))
    assert results[-1].finish_reason == "proposal_shortfall"
    assert results[-1].metadata["proposal_shortfall"] is True


def test_main_attempt_budget_is_386_parents_and_eight_attempts_each() -> None:
    parents = tuple(
        ParentInput(parent_id=f"bace-{index:03d}", parent_smiles="CC", source_label=1)
        for index in range(386)
    )
    regimes = (
        AttemptRegime("base", 4, 7, 0.3, 0.9, 96),
        AttemptRegime("high_temperature", 4, 13, 0.7, 0.9, 96),
    )
    budget = MatchedAttemptBudget(
        parents=parents,
        regimes=regimes,
        expected_parent_count=386,
    )
    manifest = budget.to_dict()
    assert manifest["attempts_per_parent"] == 8
    assert manifest["total_attempts_per_variant"] == 386 * 8
    first_parent = budget.slots()[:8]
    assert [slot.temperature for slot in first_parent] == [0.3] * 4 + [0.7] * 4
    assert [slot.regime for slot in first_parent] == ["base"] * 4 + ["high_temperature"] * 4
    assert [slot.attempt_seed for slot in first_parent] == [7] * 4 + [13] * 4
    assert [slot.regime_attempt_index for slot in first_parent] == list(range(4)) * 2
    assert len(set(manifest["variant_slot_digests"].values())) == 1
    validate_attempt_matched_schedules(budget.variant_schedules())
    requests = budget.requests()[:8]
    assert all(request.attempt_budget_sha256 == manifest["budget_sha256"] for request in requests)
    assert [request.slot_id for request in requests] == [slot.slot_id for slot in first_parent]

    tampered = dict(budget.variant_schedules())
    rows = list(tampered[LLMProposerVariant.CHEMLLM_SFT.value])
    rows[0] = replace(rows[0], temperature=0.31)
    tampered[LLMProposerVariant.CHEMLLM_SFT.value] = tuple(rows)
    with pytest.raises(LLMAblationContractError, match="do not match"):
        validate_attempt_matched_schedules(tampered)


def test_proposal_schema_and_novelty_have_no_fake_downstream_results() -> None:
    vocabulary = _vocabulary()
    request = _request()
    in_vocab = ProposalResult(
        variant=LLMProposerVariant.BRICS_FIXED,
        fragment_smiles=vocabulary.fragments[0],
        raw_text=vocabulary.fragments[0],
        finish_reason="fixed",
        metadata={},
    )
    row = build_proposal_record(
        experiment_id="experiment-1",
        request=request,
        result=in_vocab,
        vocabulary=vocabulary,
    )
    validate_proposal_record(row)
    assert row["novel_to_train_brics"] is False
    assert row["gine_evaluated"] is False
    assert row["selector_evaluated"] is False
    assert row["held_out_test_evaluated"] is False
    assert row["regime"] == "base"
    assert row["slot_id"] == request.slot_id
    assert row["attempt_budget_sha256"] == request.attempt_budget_sha256
    assert row["proposal_shortfall"] is False

    novel = dict(row, fragment_smiles="[Na+]", canonical_fragment_smiles="[Na+]", novel_to_train_brics=True)
    invalid = dict(row, fragment_smiles="not smiles", canonical_fragment_smiles=None, novel_to_train_brics=None)
    summary = summarize_novelty((row, novel, invalid), vocabulary=vocabulary)
    assert summary.total_proposals == 3
    assert summary.attempt_rows == 3
    assert summary.proposal_shortfall_count == 0
    assert summary.canonicalizable_proposals == 2
    assert summary.novel_count == 1
    assert summary.novelty_rate == 0.5

    template = proposal_output_template()
    assert template["fragment_smiles"] is None
    manifest = run_manifest_template()
    assert manifest["science_started"] is False
    assert manifest["candidate_metrics"] is None

    shortfall_result = ProposalResult(
        variant=LLMProposerVariant.BRICS_FIXED,
        fragment_smiles="",
        raw_text="",
        finish_reason="proposal_shortfall",
        metadata={"proposal_shortfall": True},
    )
    shortfall = build_proposal_record(
        experiment_id="experiment-1",
        request=_request(attempt_index=7, regime_attempt_index=3),
        result=shortfall_result,
        vocabulary=vocabulary,
    )
    shortfall_summary = summarize_novelty((shortfall,), vocabulary=vocabulary)
    assert shortfall["proposal_shortfall"] is True
    assert shortfall_summary.attempt_rows == 1
    assert shortfall_summary.total_proposals == 0
    assert shortfall_summary.proposal_shortfall_count == 1
    assert shortfall_summary.invalid_proposals == 0


def test_common_downstream_is_fixed_and_non_executing(tmp_path: Path) -> None:
    paths = {}
    for name in (
        "train_split",
        "validation_split",
        "calibration_split",
        "test_split",
        "gine_checkpoint",
        "temperature_manifest",
        "feature_schema",
        "selector_config",
        "threshold_config",
        "evaluator_config",
        "molclr_checkpoint",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = {"path": str(path), "sha256": artifact_sha256(path)}
    payload = {
        "dataset": "BACE",
        "source_label": 1,
        "num_classes": 2,
        "counterfactual_mode": "strict_flip",
        "k_max": 20,
        "table2_k": 10,
        **paths,
    }
    plan = build_common_downstream_plan(payload)
    document = plan.to_dict()
    assert document["shared_identity_for_all_variants"] is True
    assert document["generation_access"] == ["train"]
    assert document["generation_oracle_ranking"] is False
    assert document["selector_access"] == ["calibration"]
    assert document["test_access_after_selector_freeze_only"] is True
    assert document["science_execution_requested"] is False
    assert document["science_executed"] is False
    with pytest.raises(LLMAblationContractError, match="construct/validate"):
        build_common_downstream_plan(payload, execute_science=True)


def _bound_framework_fixture(
    tmp_path: Path,
    *,
    cohort_label: int = 1,
) -> tuple[dict, Path, Path]:
    cohort = tmp_path / "bace_train_parents.csv"
    with cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("molecule_id", "smiles", "split", "label"),
        )
        writer.writeheader()
        for index in range(386):
            writer.writerow(
                {
                    "molecule_id": f"bace-{index:03d}",
                    "smiles": "CCOC(=O)NCC",
                    "split": "train",
                    "label": cohort_label,
                }
            )

    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"tokenizer")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    gine = oracle / "model.pt"
    feature = oracle / "feature_schema.json"
    temperature = oracle / "temperature_scaling.json"
    for path, content in (
        (gine, b"gine"),
        (feature, b"{}\n"),
        (temperature, b'{"temperature":1.0}\n'),
    ):
        path.write_bytes(content)
    split_paths = {}
    for split in ("train", "validation", "calibration", "test"):
        path = tmp_path / f"{split}.csv"
        path.write_text(f"split\n{split}\n", encoding="utf-8")
        split_paths[split] = path
    selector_root = tmp_path / "selector"
    selector_manifest = _write_json(selector_root / "frozen_selection_manifest.json", {"K": 20})
    thresholds = _write_json(selector_root / "thresholds.json", {"values": []})
    variant_configs = _write_json(selector_root / "variant_configs.json", {"fixed": True})
    final_root = tmp_path / "final"
    evaluator = _write_json(final_root / "evaluation_manifest.json", {"status": "PASS"})
    molclr = tmp_path / "molclr.pt"
    molclr.write_bytes(b"molclr")

    proposal_contract = {
        "main_proposer_lineage": "CHEMLLM_BASE_FRESH_LORA_PPO",
        "base_model_path": str(base_model),
        "base_model_sha": artifact_sha256(base_model),
        "tokenizer": {"path": str(tokenizer), "sha256": artifact_sha256(tokenizer)},
        "proposal_parent_cohort": {
            "path": str(cohort),
            "sha256": artifact_sha256(cohort),
        },
        "proposal_parent_count": 386,
        "candidate_attempts_per_parent": 8,
        "base_sampling_count": 4,
        "high_temperature_sampling_count": 4,
        "base_sampling": {
            "num_return_sequences": 4,
            "seed": 7,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
        "high_temperature_sampling": {
            "num_return_sequences": 4,
            "seed": 13,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
    }
    reference = {
        "schema_version": "bace_ours_main_reference_v1",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": 1,
        "main_final_root": str(final_root),
        "gine_checkpoint": str(gine),
        "gine_checkpoint_root": str(oracle),
        "gine_checkpoint_sha": artifact_sha256(gine),
        "temperature_sha": artifact_sha256(temperature),
        "feature_schema": str(feature),
        "feature_schema_sha": artifact_sha256(feature),
        "molclr_root": str(molclr),
        "molclr_sha": artifact_sha256(molclr),
        "dataset_split_paths": {key: str(value) for key, value in split_paths.items()},
        "dataset_split_hashes": {
            key: artifact_sha256(value) for key, value in split_paths.items()
        },
        "proposal_contract": proposal_contract,
        "selector_contract": {
            "selector_manifest": {
                "path": str(selector_manifest),
                "sha256": artifact_sha256(selector_manifest),
            },
            "thresholds": {
                "path": str(thresholds),
                "sha256": artifact_sha256(thresholds),
            },
            "variant_configs": {
                "path": str(variant_configs),
                "sha256": artifact_sha256(variant_configs),
            },
        },
        "evaluation_config_sha": artifact_sha256(evaluator),
        "llm_variant_availability": {
            "BRICS_FIXED": {"status": "CONFIG_ONLY", "checkpoint": None},
            "CHEMLLM_PRETRAINED": {
                "status": AVAILABLE,
                "checkpoint": str(base_model),
                "checkpoint_sha": artifact_sha256(base_model),
            },
            "CHEMLLM_SFT": {
                "status": BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
            },
            "CHEMLLM_SFT_PPO": {
                "status": BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
            },
        },
        "matched_sft_checkpoint_available": False,
    }
    reference["reference_contract_sha256"] = canonical_json_sha256(reference)
    reference_path = _write_json(tmp_path / "bace_main_reference.json", reference)

    config = json.loads(
        (
            REPO_ROOT / "configs/ablations/llm/llm_proposer_ablation_v1.template.json"
        ).read_text(encoding="utf-8")
    )
    config["main_reference"] = {
        "path": str(reference_path),
        "sha256": artifact_sha256(reference_path),
    }
    config["train_input"].update(
        {"path": str(cohort), "sha256": artifact_sha256(cohort)}
    )
    config["generator_assets"]["CHEMLLM_PRETRAINED"].update(
        {
            "base_model": {
                "path": str(base_model),
                "sha256": artifact_sha256(base_model),
            },
            "tokenizer": {
                "path": str(tokenizer),
                "sha256": artifact_sha256(tokenizer),
            },
        }
    )
    pin_paths = {
        "train_split": split_paths["train"],
        "validation_split": split_paths["validation"],
        "calibration_split": split_paths["calibration"],
        "test_split": split_paths["test"],
        "gine_checkpoint": gine,
        "temperature_manifest": temperature,
        "feature_schema": feature,
        "selector_config": variant_configs,
        "threshold_config": thresholds,
        "evaluator_config": evaluator,
        "molclr_checkpoint": molclr,
    }
    for name, path in pin_paths.items():
        config["common_downstream"][name] = {
            "path": str(path),
            "sha256": artifact_sha256(path),
        }
    spec_path = _write_json(tmp_path / "ablation.json", config)
    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text("seed: 7\n", encoding="utf-8")
    return config, spec_path, runtime_config


def test_builder_binds_actual_main_reference_and_rejects_contract_drift(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    config, spec_path, runtime_config = _bound_framework_fixture(tmp_path)
    payloads = build_framework_payloads(
        spec=config,
        spec_path=spec_path,
        runtime_config_path=runtime_config,
    )
    binding = payloads["bace_main_reference_binding.json"]
    assert binding["proposal_parent_count"] == 386
    assert binding["candidate_attempts_per_parent"] == 8
    assert binding["generator_assets_bound"] is True
    assert binding["common_downstream_bound"] is True

    tampered_budget = copy.deepcopy(config)
    tampered_budget["attempt_budget"]["regimes"][0]["seed"] = 8
    with pytest.raises(LLMAblationContractError, match="attempt_budget must equal"):
        build_framework_payloads(
            spec=tampered_budget,
            spec_path=spec_path,
            runtime_config_path=runtime_config,
        )

    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter.bin").write_bytes(b"sft")
    tampered_availability = copy.deepcopy(config)
    tampered_availability["generator_assets"]["CHEMLLM_SFT"] = {
        "availability": AVAILABLE,
        "blocker_reason": None,
        "base_model": copy.deepcopy(
            config["generator_assets"]["CHEMLLM_PRETRAINED"]["base_model"]
        ),
        "tokenizer": copy.deepcopy(
            config["generator_assets"]["CHEMLLM_PRETRAINED"]["tokenizer"]
        ),
        "sft_adapter": {"path": str(sft), "sha256": artifact_sha256(sft)},
        "ppo_adapter": None,
    }
    with pytest.raises(LLMAblationContractError, match="does not match BACE main reference"):
        build_framework_payloads(
            spec=tampered_availability,
            spec_path=spec_path,
            runtime_config_path=runtime_config,
        )

    alternate_selector = _write_json(tmp_path / "alternate_selector.json", {"K": 20})
    tampered_downstream = copy.deepcopy(config)
    tampered_downstream["common_downstream"]["selector_config"] = {
        "path": str(alternate_selector),
        "sha256": artifact_sha256(alternate_selector),
    }
    with pytest.raises(LLMAblationContractError, match="not bound"):
        build_framework_payloads(
            spec=tampered_downstream,
            spec_path=spec_path,
            runtime_config_path=runtime_config,
        )


def test_builder_rejects_non_source_label_one_parent_cohort(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    config, spec_path, runtime_config = _bound_framework_fixture(
        tmp_path,
        cohort_label=0,
    )
    with pytest.raises(LLMAblationContractError, match="source_label=1"):
        build_framework_payloads(
            spec=config,
            spec_path=spec_path,
            runtime_config_path=runtime_config,
        )


def test_config_and_paired_slurm_freeze_main_contract() -> None:
    config_path = (
        REPO_ROOT / "configs/ablations/llm/llm_proposer_ablation_v1.template.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["main_reference"] == {"path": None, "sha256": None}
    assert config["main_proposal_reference"] == {
        "policy_lineage": "CHEMLLM_BASE_PLUS_FRESH_LORA_PLUS_PPO",
        "matched_independent_sft_checkpoint_available": False,
        "sft_variants_state": BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
        "train_parent_count": 386,
        "attempts_per_parent": 8,
        "science_reference_only": True,
    }
    assert config["attempt_budget"]["expected_parent_count"] == 386
    assert config["attempt_budget"]["expected_attempts_per_parent"] == 8
    assert config["generator_assets"]["CHEMLLM_PRETRAINED"]["tokenizer"] == {
        "path": None,
        "sha256": None,
    }
    assert config["attempt_budget"]["regimes"] == [
        {
            "name": "base",
            "attempts_per_parent": 4,
            "seed": 7,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
        {
            "name": "high_temperature",
            "attempts_per_parent": 4,
            "seed": 13,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
    ]

    slurm = (REPO_ROOT / "scripts/slurm/build_llm_ablation_plan.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert required in slurm
