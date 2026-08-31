from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autodl import verify_tastemolnet_ours_ppo_smoke as verifier
from scripts import train_tastemolnet_gnn_ppo as runner


def _science_evidence(*, steps: int = 5) -> dict[str, object]:
    return {
        "status": "PASS",
        "stage": "T6_OURS_SMOKE",
        "optimizer_step_count": steps,
        "gate_sha256": "a" * 64,
        "output_inventory_sha256": "b" * 64,
    }


def test_independent_verifier_publishes_fresh_receipt_without_mutating_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    science = tmp_path / "science"
    science.mkdir()
    sentinel = science / "PASS"
    sentinel.write_text("science\n", encoding="utf-8")
    before = sentinel.read_bytes()
    monkeypatch.setattr(
        verifier, "validate_taste_ppo_output", lambda root: _science_evidence()
    )
    monkeypatch.setattr(verifier, "_rename_noreplace", os.rename)
    receipt = tmp_path / "verification"
    evidence = verifier.verify(science, receipt)
    assert evidence["independent_verifier"] is True
    assert sentinel.read_bytes() == before
    assert (receipt / "PASS").read_text(encoding="utf-8") == verifier.MARKER + "\n"
    gate = json.loads((receipt / "gate.json").read_text(encoding="utf-8"))
    assert gate["science_gate_sha256"] == "a" * 64
    assert gate["science_output_inventory_sha256"] == "b" * 64
    with pytest.raises(FileExistsError, match="must be fresh"):
        verifier.verify(science, receipt)


def test_independent_verifier_rejects_less_than_five_real_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        verifier,
        "validate_taste_ppo_output",
        lambda _root: _science_evidence(steps=4),
    )
    with pytest.raises(ValueError, match="did not return PASS"):
        verifier.verify(tmp_path / "science", tmp_path / "verification")
    assert not (tmp_path / "verification").exists()


def test_zero_step_materializer_uses_same_seed_and_frozen_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []

    class Parameter:
        def __init__(self) -> None:
            self.requires_grad = True

    class Model:
        def __init__(self, config: object) -> None:
            self.peft_config = {"default": config}
            self._parameters = [Parameter(), Parameter()]
            self.eval_called = False

        def parameters(self):
            return iter(self._parameters)

        def eval(self) -> None:
            self.eval_called = True

    peft = SimpleNamespace(
        LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=lambda _base, config: Model(config),
    )
    monkeypatch.setitem(__import__("sys").modules, "peft", peft)
    monkeypatch.setattr(
        runner,
        "build_quantized_base_model",
        lambda *_args, **_kwargs: SimpleNamespace(peft_config=None),
    )
    deps = {"set_seed": seeds.append}
    policy = runner._build_zero_step_lora_model(
        deps,
        model_path=Path("/held/base"),
        lexical_model_path=Path("/lexical/base"),
        seed=7,
        is_trainable=True,
    )
    reference = runner._build_zero_step_lora_model(
        deps,
        model_path=Path("/held/base"),
        lexical_model_path=Path("/lexical/base"),
        seed=7,
        is_trainable=False,
    )
    assert seeds == [7, 7]
    assert policy.peft_config["default"].r == 8
    assert policy.peft_config["default"].target_modules == [
        "wqkv", "wo", "w1", "w2", "w3"
    ]
    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert reference.eval_called is True


def _binding_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, bytes]]:
    split_manifest = {
        "schema_version": "molecular_gnn_split_manifest_v1",
        "dataset": "tastemolnet",
        "roles": {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        },
        "files": {
            "train": {"path": "/private/train.csv", "sha256": "1" * 64},
            "validation": {"path": "/private/validation.csv", "sha256": "2" * 64},
            "calibration": {"path": "/private/calibration.csv", "sha256": "3" * 64},
            "test": {"path": "/private/test.csv", "sha256": "4" * 64},
        },
        "train_manifest": {
            "schema_version": "molecular_graph_dataset_v1",
            "num_records": 3,
            "num_classes": 3,
            "label_counts": {"0": 1, "1": 1, "2": 1},
            "split_counts": {"train": 3},
            "source_path": "/private/train.csv",
            "source_sha256": "1" * 64,
            "dataset_fingerprint": "5" * 64,
            "feature_schema_sha256": "6" * 64,
        },
        "validation_manifest": {"schema_version": "molecular_graph_dataset_v1"},
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_evaluated_during_training": False,
        "test_used_for_checkpoint_selection": False,
    }
    frozen = {
        "checkpoint_id": "a" * 64,
        "t3_verification_sha256": "b" * 64,
        "t4_verification_sha256": "c" * 64,
        "t2_adoption_binding": {
            "receipt_sha256": "d" * 64,
            "gate_sha256": "e" * 64,
            "formal_bundle_model_sha256": "a" * 64,
        },
        "t4_science": {
            "checkpoint_id": "a" * 64,
            "feature_schema_sha256": "6" * 64,
            "train_payload_loaded": False,
            "validation_payload_loaded": False,
            "test_payload_loaded": False,
            "strict_flip_count": 38,
            "distinct_flipped_parent_count": 17,
        },
    }
    t5 = {
        "source_model_inventory_sha256": "f" * 64,
        "source_adapter_present": False,
    }
    managed_t5 = {"verification_sha256": "9" * 64}
    payloads = {
        "split_manifest.json": (json.dumps(split_manifest) + "\n").encode("utf-8"),
        "feature_schema.json": (
            json.dumps({"schema_sha256": "6" * 64}) + "\n"
        ).encode("utf-8"),
        "temperature_scaling.json": (
            json.dumps(
                {
                    "status": "fit",
                    "temperature": 1.9724769811,
                    "selection_split": "validation",
                    "test_used_for_fit": False,
                }
            )
            + "\n"
        ).encode("utf-8"),
    }
    return frozen, t5, managed_t5, payloads


def test_managed_v2_binding_normalizes_real_t2_t3_t4_t5_shapes() -> None:
    frozen, t5, managed_t5, payloads = _binding_payloads()
    binding = runner._build_taste_managed_binding_v2(
        frozen_oracle=frozen,
        t5_evidence=t5,
        managed_t5_evidence=managed_t5,
        checkpoint_payloads=payloads,
    )
    assert binding.evidence() == {
        "dataset": "tastemolnet",
        "num_classes": 3,
        "source_label": 1,
        "t2_receipt_sha": "d" * 64,
        "t2_model_sha": "a" * 64,
        "t2_feature_schema_sha": "6" * 64,
        "t2_dataset_sha": "5" * 64,
        "t2_split_sha": runner.hashlib.sha256(payloads["split_manifest.json"]).hexdigest(),
        "t3_receipt_sha": "b" * 64,
        "t3_predecessor_t2_sha": "e" * 64,
        "t3_calibrated_model_sha": "a" * 64,
        "t3_temperature": 1.9724769811,
        "t3_feature_schema_sha": "6" * 64,
        "t3_validation_split_sha": "2" * 64,
        "t3_test_loaded": False,
        "t4_receipt_sha": "c" * 64,
        "t4_predecessor_t3_sha": "b" * 64,
        "t4_calibration_split_sha": "3" * 64,
        "t4_gine_sha": "a" * 64,
        "t4_test_loaded": False,
        "t4_strict_flip_count": 38,
        "t4_flipped_parent_count": 17,
        "t5_receipt_sha": "9" * 64,
        "t5_clean_base_sha": "f" * 64,
        "t5_train_only": True,
        "t5_validation_loaded": False,
        "t5_calibration_loaded": False,
        "t5_test_loaded": False,
    }


def test_managed_v2_binding_reports_field_level_drift() -> None:
    frozen, t5, managed_t5, payloads = _binding_payloads()
    frozen = {
        **frozen,
        "t4_science": {
            **frozen["t4_science"],
            "feature_schema_sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="t4_feature_schema_sha"):
        runner._build_taste_managed_binding_v2(
            frozen_oracle=frozen,
            t5_evidence=t5,
            managed_t5_evidence=managed_t5,
            checkpoint_payloads=payloads,
        )


def test_t6_source_adopts_managed_t5_and_keeps_frozen_oracle_binding() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "hold_verified_managed_final" in source
    assert "ADOPTED_CLEAN_GENERIC_BASE" in source
    assert "T6_RUNTIME_IN_MEMORY_ZERO_STEP_LORA" in source
    assert "frozen_oracle_identity" in source
    assert "TasteManagedEvidenceBindingV2" in source
    assert "_build_taste_managed_binding_v2(" in source
    wrapper = (
        Path(runner.__file__).parents[1]
        / "scripts/autodl/run_tastemolnet_ours_ppo_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "TASTEMOLNET_T3_CHECKPOINT" in wrapper
    assert '--gnn-checkpoint "$TASTEMOLNET_T3_CHECKPOINT"' in wrapper
