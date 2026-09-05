from __future__ import annotations

import json
import random
from pathlib import Path
import importlib.util
from datetime import datetime, timezone, timedelta
import fcntl
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ablations.llm.bace_native_runtime import render_native_inputs, run_generation, validate_calls
from src.ablations.llm.bace_readiness import ppo_adoption_decision
from src.ablations.llm.parameter_count import count_actual_loaded_parameters


def calls():
    return [{"parent_id": f"p{i}", "parent_smiles": "CCO", "shard_id": 0,
             "regime": regime, "seed": seed, "temperature": temp, "prompt": "FRAGMENT_SMILES:"}
            for regime, seed, temp in (("base", 7, .3), ("high_temperature", 13, .7))
            for i in range(3)]


class TinyRealTorchRuntime:
    """No model weights: exercise all three actual RNG streams and serialization."""
    def generate_call(self, call):
        return [{"fragment_smiles": "CC", "raw_text": repr((random.random(), float(np.random.rand()),
                  float(torch.rand(())))), "candidate_index": i} for i in range(4)]


def test_exact_generation_rng_resume_and_four_sequence_budget(tmp_path):
    spec = {"schema_version": "bace_native_llm_task_v1", "variant": "CHEMLLM_7B_OFF_THE_SHELF", "calls": calls()}
    a, b = tmp_path / "continuous", tmp_path / "resumed"
    full = run_generation(spec=spec, output_root=a, runtime=TinyRealTorchRuntime())
    partial = run_generation(spec=spec, output_root=b, runtime=TinyRealTorchRuntime(), max_calls=2)
    assert partial["status"] == "PAUSED_AT_CALL_CHECKPOINT"
    random.seed(900); np.random.seed(900); torch.manual_seed(900)
    resumed = run_generation(spec=spec, output_root=b, runtime=TinyRealTorchRuntime(), resume=True)
    assert resumed["status"] == full["status"] == "CANDIDATE_POOL_PASS"
    assert full["proposal_attempts"] == 24
    assert (a / "candidate_pool.jsonl").read_bytes() == (b / "candidate_pool.jsonl").read_bytes()
    assert len(list(b.glob("call-*.pt"))) == 2
    assert resumed["safe_pause_bound_seconds"] is None
    assert not resumed["training_performed"]


def test_resume_refuses_changed_scientific_contract(tmp_path):
    spec = {"schema_version": "bace_native_llm_task_v1", "variant": "CHEMLLM_7B_OFF_THE_SHELF", "calls": calls()}
    run_generation(spec=spec, output_root=tmp_path / "run", runtime=TinyRealTorchRuntime(), max_calls=1)
    changed = {**spec, "additional_scientific_setting": True}
    with pytest.raises(ValueError, match="resume contract"):
        run_generation(spec=changed, output_root=tmp_path / "run", resume=True, runtime=TinyRealTorchRuntime())


def test_no_per_parent_seed_reset_group_interleaving():
    original = calls()
    interleaved = [original[0], original[3], original[1], original[4], original[2], original[5]]
    with pytest.raises(ValueError, match="contiguous"):
        validate_calls(interleaved)


def test_native_prompt_dispatch_does_not_use_generic_tokenizer_template():
    observed = []
    model = SimpleNamespace(build_inputs=lambda tokenizer, prompt, **kw: observed.append(kw) or {"prompt": prompt})
    wrapped = SimpleNamespace(get_base_model=lambda: model)
    assert render_native_inputs(wrapped, object(), "same task", "7b") == {"prompt": "same task"}
    render_native_inputs(model, object(), "same task", "2b")
    assert observed == [{"history": []}, {"history": [], "meta_instruction": ""}]


def test_main_plain_prompt_forbids_candidate_adoption_but_reuses_ppo():
    result = ppo_adoption_decision({"prompt": {"rendering": "plain_text_no_tokenizer_chat_template"},
        "ppo": {"checkpoint_root": "/existing", "optimizer_updates": 300}}, rendering="PINNED_NATIVE_MODEL_BUILD_INPUTS")
    assert result["state"] == "MATCHED_REGEN_REQUIRED"
    assert result["checkpoint_reused"] and not result["training_required"]
    assert not result["project_sft_checkpoint_exists"]


def test_quantized_parameter_count_uses_loaded_logical_shape_not_packed_bytes():
    parameter = SimpleNamespace(numel=lambda: 8, element_size=lambda: 1,
        quant_state=SimpleNamespace(shape=(4, 4)), dtype="uint8", requires_grad=False)
    model = SimpleNamespace(named_parameters=lambda: [("linear.weight", parameter)])
    result = count_actual_loaded_parameters(model)
    assert result.total_parameters == 16
    assert result.weight_bytes == 8


def test_checkpoint_byte_corruption_fails_before_model_load(tmp_path):
    spec = {"schema_version": "bace_native_llm_task_v1", "variant": "CHEMLLM_7B_OFF_THE_SHELF", "calls": calls()}
    output = tmp_path / "run"
    run_generation(spec=spec, output_root=output, runtime=TinyRealTorchRuntime(), max_calls=1)
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    (output / latest["checkpoint_file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA differs"):
        run_generation(spec=spec, output_root=output, resume=True, runtime=TinyRealTorchRuntime())


def test_main_priority_pause_before_first_call_is_resumable(tmp_path):
    spec = {"schema_version": "bace_native_llm_task_v1", "variant": "CHEMLLM_7B_OFF_THE_SHELF", "calls": calls()}
    output = tmp_path / "run"
    receipt = run_generation(spec=spec, output_root=output, runtime=TinyRealTorchRuntime(), continue_guard=lambda: False)
    assert receipt["next_call"] == 0
    assert (output / "call-000000.pt").is_file()
    assert run_generation(spec=spec, output_root=output, runtime=TinyRealTorchRuntime(), resume=True)["status"] == "CANDIDATE_POOL_PASS"


def test_cli_requires_actual_held_existing_lease_and_rejects_borrow(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/ablations/llm/run_bace_native_llm.py"
    module_spec = importlib.util.spec_from_file_location("native_cli", path)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    lock_path = tmp_path / "gpu-GPU-fixture.lock"
    with lock_path.open("w+") as held:
        held.write(json.dumps({"state": "LOCKED", "gpu_uuid": "GPU-fixture", "gpu_index": 0, "pid": os.getpid()}))
        held.flush()
        evidence = {"observed_at": datetime.now(timezone.utc).isoformat(), "gpu_lease_mode": "EXCLUSIVE_IDLE",
            "gpu_lock_path": str(lock_path), "gpu_uuid": "GPU-fixture", "gpu_index": 0,
            "gpu_owner_pid": os.getpid(), "target_gpu_uuid": "GPU-fixture"}
        # Historical single-FD/no-child/no-single-slot evidence is no longer
        # dispatchable, even when its UUID lock really is held. Real owner to
        # child/grandchild transport is exercised by the CPU subprocess suite.
        with pytest.raises(ValueError, match="INVALID_OWNER_EVIDENCE"):
            cli.resource_gate(evidence, held.fileno())
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="INVALID_OWNER_EVIDENCE"):
            cli.resource_gate(evidence, held.fileno())
        with pytest.raises(ValueError):
            cli.resource_gate({**evidence, "gpu_lease_mode": "BORROW"}, held.fileno())
        old = (datetime.now(timezone.utc) - timedelta(seconds=121)).isoformat()
        with pytest.raises(ValueError, match="STALE"):
            cli.resource_gate({**evidence, "observed_at": old}, held.fileno())
