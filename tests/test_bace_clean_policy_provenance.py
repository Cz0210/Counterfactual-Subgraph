from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.train.bace_policy_init import (
    InitializerClassification,
    audit_policy_initializer,
    finalize_adapter_manifest,
    select_policy_initializer,
    sha256_file,
    source_model_hash_from_passed_audit,
    validate_frozen_train_contract,
    validate_policy_provenance_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adapter(path: Path) -> None:
    _write_json(path / "adapter_config.json", {"peft_type": "LORA", "r": 8})
    (path / "adapter_model.safetensors").write_bytes(b"frozen-adapter-bytes")


def test_initializer_audit_is_fail_closed_and_detects_rf(tmp_path: Path) -> None:
    missing = audit_policy_initializer(tmp_path / "missing")
    assert missing.classification == InitializerClassification.MISSING.value
    assert missing.eligible is False

    unknown = tmp_path / "checkpoint-3"
    _adapter(unknown)
    audited_unknown = audit_policy_initializer(unknown, content_hash=False)
    assert audited_unknown.classification == InitializerClassification.UNKNOWN.value
    assert audited_unknown.eligible is False

    contaminated = tmp_path / "rf-ranked-sft"
    _adapter(contaminated)
    _write_json(
        contaminated / "policy_provenance.json",
        {"policy_initialization_type": "oracle_neutral_sft", "teacher_backend": "rf"},
    )
    audited_rf = audit_policy_initializer(contaminated, content_hash=False)
    assert audited_rf.classification == InitializerClassification.RF_CONTAMINATED.value
    assert audited_rf.eligible is False


def test_raw_chemllm_base_and_explicit_clean_adapter_are_eligible(
    tmp_path: Path,
) -> None:
    base = tmp_path / "ChemLLM-7B-Chat"
    _write_json(base / "config.json", {"model_type": "internlm2"})
    (base / "model-00001-of-00001.safetensors").write_bytes(b"base")
    audited_base = audit_policy_initializer(
        base, kind_hint="raw_base", content_hash=False
    )
    assert audited_base.classification == InitializerClassification.CLEAN_CHEMLLM_BASE.value
    assert audited_base.eligible is True

    adapter = tmp_path / "clean" / "adapter"
    _adapter(adapter)
    manifest = finalize_adapter_manifest(
        adapter_dir=adapter,
        manifest={
            "policy_initialization_type": "oracle_neutral_sft",
            "dataset": "bace",
            "data_split_used": "train_only",
            "rf_reference_count": 0,
            "gnn_reward_used": False,
            "formal_validation_loaded": False,
            "policy_internal_validation_loaded": True,
            "policy_internal_validation_source": "train_only",
            "calibration_loaded": False,
            "test_loaded": False,
            "source_model_hash": "source-sha",
            "training_data_hash": "training-sha",
        },
        output_path=tmp_path / "clean" / "policy_provenance.json",
    )
    audited_adapter = audit_policy_initializer(adapter, content_hash=False)
    assert audited_adapter.classification == (
        InitializerClassification.CLEAN_ORACLE_NEUTRAL_SFT.value
    )
    assert audited_adapter.eligible is True
    validated = validate_policy_provenance_manifest(
        adapter, tmp_path / "clean" / "policy_provenance.json"
    )
    assert validated["policy_initializer_hash"] == manifest["policy_initializer_hash"]
    assert validated["rf_reference_count"] == 0


def test_clean_selection_uses_predeclared_lexicographic_gate() -> None:
    base = {
        "path": "/base",
        "classification": "CLEAN_CHEMLLM_BASE",
        "eligible": True,
        "parse_ok_rate": 0.9,
        "direct_substructure_rate": 0.7,
        "oracle_evaluable_rate": 0.8,
        "strict_flip_rate": 0.1,
        "unique_fragment_rate": 0.9,
    }
    sft = {
        **base,
        "path": "/sft",
        "classification": "CLEAN_ORACLE_NEUTRAL_SFT",
        "strict_flip_rate": 0.2,
    }
    rf = {**sft, "path": "/rf", "classification": "RF_CONTAMINATED", "eligible": False}
    assert select_policy_initializer([base, sft, rf])["path"] == "/sft"


def test_frozen_train_contract_rejects_heldout_or_path_substitution(
    tmp_path: Path,
) -> None:
    train = tmp_path / "splits" / "train.csv"
    train.parent.mkdir()
    train.write_text("smiles,label\nCCO,1\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    _write_json(
        checkpoint / "split_manifest.json",
        {
            "dataset": "bace",
            "roles": {
                "train": "model_fitting",
                "validation": "checkpoint_selection_and_temperature_calibration",
                "calibration": "reserved_for_threshold_and_selector_only",
                "test": "frozen_model_final_quality_evaluation",
            },
            "files": {
                "train": {"path": str(train.resolve()), "sha256": sha256_file(train)},
                "test": {"path": str((tmp_path / "test.csv").resolve()), "sha256": "heldout"},
            },
        },
    )
    contract = validate_frozen_train_contract(checkpoint, train.resolve())
    assert contract["calibration_loaded"] is False
    assert contract["test_loaded"] is False

    substitute = tmp_path / "substitute.csv"
    substitute.write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="path differs"):
        validate_frozen_train_contract(checkpoint, substitute.resolve())


def test_autodl_audit_wrapper_binds_exact_timestamped_csv_override(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    data_root = tmp_path / "persistent"
    runtime = data_root / "counterfactual-subgraph-runtime"
    output = runtime / "outputs" / "autodl" / "audit-run"
    audit_csv = (
        runtime
        / "outputs"
        / "autodl"
        / "audits"
        / "bace_policy_initializer_provenance_20260822T000000Z.csv"
    )
    base = tmp_path / "ChemLLM-7B-Chat"
    _write_json(base / "config.json", {"model_type": "internlm2"})
    (base / "model.safetensors").write_bytes(b"small-test-model")
    data_root.mkdir()
    completed = subprocess.run(
        [
            "bash",
            str(repository / "scripts" / "autodl" / "run_bace_gnn_ppo_stage.sh"),
            "BACE_POLICY_PROVENANCE_AUDIT",
        ],
        cwd=repository,
        env={
            **os.environ,
            "AUTODL_PYTHON": sys.executable,
            "AUTODL_DATA_ROOT": str(data_root),
            "AUTODL_RUNTIME_ROOT": str(runtime),
            "OUTPUT_ROOT": str(output),
            "CHEMLLM_MODEL_PATH": str(base),
            "BACE_POLICY_AUDIT_CSV": str(audit_csv),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert audit_csv.is_file()
    assert (output / "PASS").read_text(encoding="utf-8") == (
        "[BACE_POLICY_PROVENANCE_AUDIT_PASS]\n"
    )
    audit_manifest = json.loads(
        (output / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert audit_manifest["output_csv"] == str(audit_csv.resolve())
    selection_path = output / "initializer_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert source_model_hash_from_passed_audit(
        selection_path, expected_model_path=base
    ) == selection["selected"]["policy_initializer_hash"]

    selection_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection differs"):
        source_model_hash_from_passed_audit(
            selection_path, expected_model_path=base
        )
