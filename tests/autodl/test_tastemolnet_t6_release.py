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


def test_t6_source_adopts_managed_t5_and_keeps_frozen_oracle_binding() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "hold_verified_managed_final" in source
    assert "ADOPTED_CLEAN_GENERIC_BASE" in source
    assert "T6_RUNTIME_IN_MEMORY_ZERO_STEP_LORA" in source
    assert "frozen_oracle_identity" in source
