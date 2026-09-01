from __future__ import annotations

import hashlib
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.ablations import main_reference
from src.ablations.contracts import ContractError
from src.ablations.main_reference import (
    BaceOursReferenceInputs,
    build_bace_ours_main_reference,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bace_main_reference_is_evidence_driven_and_blocks_unmatched_sft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "final"
    authority_state = _write(tmp_path / "authority-state.json", {"schema_version": "fixture"})
    monkeypatch.setattr(
        main_reference,
        "validate_matrix_authority_pointer",
        lambda _state: {
            "exact_16_cells": False,
            "applied_cells": ["BACE/Ours"],
            "root": str((tmp_path / "authority").resolve()),
            "matrix_status_sha256": "8" * 64,
            "combined_audit_sha256": "9" * 64,
            "cell_roots": {"BACE/Ours": str(final.resolve())},
        },
    )
    oracle = tmp_path / "oracle"
    ppo = tmp_path / "ppo"
    selector_root = tmp_path / "selector"
    model = _write(oracle / "model.pt", b"gine")
    schema = _write(oracle / "feature_schema.json", {"schema": 1})
    temperature = _write(oracle / "temperature_scaling.json", {"temperature": 1.5})
    split_files = {}
    for split in ("train", "validation", "calibration", "test"):
        path = _write(tmp_path / f"{split}.csv", f"{split}\n".encode())
        split_files[split] = {"path": str(path), "sha256": _sha(path)}
    _write(oracle / "split_manifest.json", {"files": split_files})
    _write(
        oracle / "model_card.json",
        {"checkpoint_id": _sha(model), "training_commit": "train-commit"},
    )

    molclr = _write(tmp_path / "molclr.pth", b"molclr")
    final_run = {
        "status": "PASS",
        "dataset": "BACE",
        "method": "Ours",
        "source_label": 1,
        "rf_oracle_used": False,
        "cf_mode": "strict_flip",
        "oracle_checkpoint_hash": _sha(model),
        "molclr_checkpoint_hash": _sha(molclr),
    }
    _write(final / "run_manifest.json", final_run)
    _write(
        final / "evaluation_manifest.json",
        {
            "status": "PASS",
            "test_used_for_selection": False,
            "selection_frozen_before_test": True,
            "threshold_config_hash": "a" * 64,
        },
    )
    _write(final / "summary.json", {"table2_k": 10})
    _write(final / "final_artifact_audit.json", {"status": "PASS"})

    policy_weights = _write(ppo / "adapter_model.safetensors", b"ppo")
    policy_config = _write(ppo / "adapter_config.json", {"adapter": 1})
    _write(ppo / "tokenizer.model", b"tokenizer")
    policy_hash = "b" * 64
    _write(ppo / "ppo_manifest.json", {"status": "PASS", "policy_checkpoint_hash": policy_hash})
    _write(
        ppo / "run_manifest.json",
        {
            "git_commit": "a" * 40,
            "args": {
                "include_label_in_prompt": True,
                "core_output_mode": True,
                "enable_parent_projection": False,
                "projection_min_atoms": 3,
                "enable_minimal_syntax_repair": True,
                "repair_max_edits": 4,
                "enable_component_salvage": True,
                "component_salvage_method": "largest_then_best_parent_match",
                "component_salvage_min_atoms": 3,
            },
        },
    )
    _write(
        ppo / "policy_provenance.json",
        {
            "adapter_initialized_from_scratch": True,
            "data_split_used": "none",
            "training_data_hash": None,
            "source_model_path": "/models/ChemLLM-7B-Chat",
            "source_model_hash": "c" * 64,
            "adapter_dir": "/clean/initializer",
            "policy_initializer_hash": "d" * 64,
        },
    )

    cohort_sha = "e" * 64
    prep = _write(
        tmp_path / "prep.json",
        {
            "status": "PASS",
            "fixed_shards": {
                "train": {
                    "parent_ids_sha256": cohort_sha,
                    "parent_count": 386,
                    "parent_manifest_identity": {"path": "/parents", "sha256": "f" * 64},
                }
            },
        },
    )
    policy_identity = {
        "adapter_weights": {"sha256": _sha(policy_weights)},
        "adapter_config": {"sha256": _sha(policy_config)},
    }
    base = _write(
        tmp_path / "base.json",
        {
            "status": "PASS",
            "all_parent_ids_sha256": cohort_sha,
            "policy_checkpoint_hash": policy_hash,
            "policy_identity": policy_identity,
            "generation_config": {"num_return_sequences": 4, "seed": 7, "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 96},
        },
    )
    high = _write(
        tmp_path / "high.json",
        {
            "status": "PASS",
            "all_parent_ids_sha256": cohort_sha,
            "policy_checkpoint_hash": policy_hash,
            "policy_identity": policy_identity,
            "generation_config": {"num_return_sequences": 4, "seed": 13, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 96},
        },
    )
    merged = _write(
        tmp_path / "merged.json",
        {
            "status": "PASS",
            "policy_checkpoint_hash": policy_hash,
            "deterministic_merge": True,
            "input_row_count": 24,
            "merged_row_count": 12,
            "candidate_universe_count": 7,
            "candidate_universe_hash": "1" * 64,
            "candidate_pool_hash": "2" * 64,
        },
    )
    verification = _write(
        tmp_path / "verification.json",
        {
            "status": "PASS",
            "distance_type": "node_wasserstein",
            "match_selection_policy": "minimum",
            "no_valid_strict_flip_semantics": "+inf",
            "distance_provider_stats": {
                "distance_line": "MolCLR-Node-Wasserstein",
                "distance_namespace": "fixed",
                "feature_cost": "cosine",
                "node_mass": "uniform",
                "size_penalty_beta": 0.0,
                "solver": "exact_emd2",
            },
        },
    )
    selector = _write(
        selector_root / "frozen_selection_manifest.json",
        {
            "status": "FROZEN",
            "stage": "B12_SELECTOR",
            "calibration_loaded": True,
            "selector_fitted_on_calibration": True,
            "selection_frozen": True,
            "test_loaded": False,
            "K": 20,
            "cf_mode": "strict_flip",
            "classifier_type": "gnn",
            "oracle_backend": "gnn",
            "source_label": 1,
            "num_classes": 2,
            "ordered_rule_ids": [f"rule-{index}" for index in range(20)],
            "ordered_rule_ids_sha256": "3" * 64,
            "calibration_input_hash": "4" * 64,
            "policy_checkpoint_hash": policy_hash,
            "candidate_pool_hash": "2" * 64,
            "oracle_checkpoint_hash": _sha(model),
            "molclr_checkpoint_hash": _sha(molclr),
        },
    )
    _write(selector_root / "thresholds.json", {"values": []})
    _write(selector_root / "variant_configs.json", {"fixed": True})

    result = build_bace_ours_main_reference(
        BaceOursReferenceInputs(
            matrix_authority_state=authority_state,
            final_root=final,
            oracle_root=oracle,
            ppo_root=ppo,
            train_parent_prep_manifest=prep,
            base_pool_manifest=base,
            high_temperature_pool_manifest=high,
            merged_pool_manifest=merged,
            verification_manifest=verification,
            selector_manifest=selector,
            molclr_checkpoint=molclr,
        )
    )
    assert result["status"] == "PASS"
    assert result["proposal_contract"]["candidate_attempts_per_parent"] == 8
    assert result["llm_variant_availability"]["CHEMLLM_SFT"]["status"] == "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
    assert result["llm_variant_availability"]["CHEMLLM_SFT_PPO"]["status"] == "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
    assert len(result["reference_contract_sha256"]) == 64
    assert result["proposal_contract"]["proposal_parent_count"] == 386

    incomplete_selector = json.loads(selector.read_text(encoding="utf-8"))
    incomplete_selector["selection_frozen"] = False
    _write(selector, incomplete_selector)
    with pytest.raises(ContractError, match="selector selection_frozen"):
        build_bace_ours_main_reference(
            BaceOursReferenceInputs(
                matrix_authority_state=authority_state,
                final_root=final,
                oracle_root=oracle,
                ppo_root=ppo,
                train_parent_prep_manifest=prep,
                base_pool_manifest=base,
                high_temperature_pool_manifest=high,
                merged_pool_manifest=merged,
                verification_manifest=verification,
                selector_manifest=selector,
                molclr_checkpoint=molclr,
            )
        )
    _write(selector, {**incomplete_selector, "selection_frozen": True})

    wrong_parent_count = deepcopy(result)
    wrong_parent_count["proposal_contract"]["proposal_parent_count"] = 3
    with pytest.raises(ContractError, match="proposal contract changed"):
        main_reference.validate_main_reference_contract(wrong_parent_count)

    wrong_sampling = deepcopy(result)
    wrong_sampling["proposal_contract"]["base_sampling"]["seed"] = 999
    with pytest.raises(ContractError, match="frozen BACE sampling contract"):
        main_reference.validate_main_reference_contract(wrong_sampling)

    invented_sft = deepcopy(result)
    invented_sft["llm_variant_availability"]["CHEMLLM_SFT"] = {
        "status": "AVAILABLE",
        "checkpoint": "/invented/sft",
        "checkpoint_sha": "f" * 64,
    }
    with pytest.raises(ContractError, match="matched-SFT blocker"):
        main_reference.validate_main_reference_contract(invented_sft)


def test_bace_reference_rejects_non_authority_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _write(tmp_path / "state.json", {})
    final = tmp_path / "final"
    final.mkdir()
    monkeypatch.setattr(
        main_reference,
        "validate_matrix_authority_pointer",
        lambda _state: {
            "exact_16_cells": False,
            "applied_cells": ["BACE/Ours"],
            "cell_roots": {"BACE/Ours": str((tmp_path / "other").resolve())},
        },
    )
    with pytest.raises(ContractError, match="unique matrix authority cell"):
        build_bace_ours_main_reference(
            BaceOursReferenceInputs(
                matrix_authority_state=state,
                final_root=final,
                oracle_root=tmp_path / "oracle",
                ppo_root=tmp_path / "ppo",
                train_parent_prep_manifest=tmp_path / "prep.json",
                base_pool_manifest=tmp_path / "base.json",
                high_temperature_pool_manifest=tmp_path / "high.json",
                merged_pool_manifest=tmp_path / "merged.json",
                verification_manifest=tmp_path / "verification.json",
                selector_manifest=tmp_path / "selector.json",
                molclr_checkpoint=tmp_path / "molclr.pt",
            )
        )
