"""Tiny CPU fixtures; CUDA inventory is mocked, never a GPU-science proof."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.ablations.llm import native_gpu_smoke as smoke
from src.ablations.llm import bace_native_runtime as runtime
from src.ablations.llm import existing_gpu_owner as owner
from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils.autodl_runtime import GPUObservation
from src.utils.final16_owner_registry_v1 import build_owner_registry


def test_dispatch_cohort_is_actual_main_true_source_not_gnn_native():
    from src.ablations.llm import bace_readiness, bace_common_downstream
    assert bace_readiness.COHORT == bace_common_downstream.COHORT
    assert bace_readiness.COHORT == "all_true_source_label_1_parents_as_main_BACE_load_bace_parents"


def test_held_empty_main_coordination_lock_is_a_blocker_not_idle(tmp_path):
    import fcntl
    fixture = helper()
    config, _ = fixture.source_fixture(tmp_path)
    gpu = GPUObservation(0, 'GPU-fixture', 'CPU fixture', 1000, 0, 1000, 0)
    Path(config['gpu_lock_root']).mkdir()
    locked = Path(config['gpu_lock_root']) / 'gpu-main.coordination.lock'
    with locked.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sampler = owner.ResourceSampler(config, 0, gpu.uuid, inventory=lambda: [gpu])
        sample = sampler.sample()
        assert not sample['owners_healthy']
        assert any('HELD_LEASE_METADATA_UNAVAILABLE' in item for item in sample['source_blockers'])
        assert locked.read_bytes() == b''


def test_declared_other_main_lease_does_not_block_unreserved_gpu(tmp_path):
    import fcntl
    config, _ = helper().source_fixture(tmp_path, reserved=True)
    Path(config['gpu_lock_root']).mkdir()
    path = Path(config['gpu_lock_root']) / 'gpu-main.coordination.lock'
    registry_path = Path(config['main_registry_path'])
    registry = json.loads(registry_path.read_text())
    registry['gpu_leases'][0]['lease_path'] = str(path)
    registry = build_owner_registry(registry_id=registry['registry_id'],
        matrix_authority_root=registry['matrix_authority_root'], tasks=registry['tasks'],
        publishers=registry['publishers'], gpu_leases=registry['gpu_leases'], check_processes=False)
    atomic_json(registry_path, registry)
    gpu = GPUObservation(0, 'GPU-fixture', 'CPU fixture', 1000, 0, 1000, 0)
    with path.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = owner.ResourceSampler(config, 0, gpu.uuid, inventory=lambda:[gpu]).sample()
        assert sample['owners_healthy'] and not sample['gpu_main_reservation']
        assert any(row.get('role') == 'HELD_DECLARED_MAIN_COORDINATION' for row in sample['source_observations'])


def helper():
    source = Path(__file__).with_name("test_llm_existing_gpu_owner.py")
    spec = importlib.util.spec_from_file_location("owner_fixtures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task():
    calls = [{"parent_id": p, "parent_smiles": "CCO", "shard_id": 0, "prompt": "fragment",
              "regime": regime, "seed": seed, "temperature": temperature}
             for regime, seed, temperature in (("base", 7, .3), ("high_temperature", 13, .7))
             for p in (1, 2)]
    return {"schema_version": "bace_native_llm_task_v1", "variant": runtime.VARIANTS[1], "calls": calls}


class TinyRuntime:
    def __init__(self, spec):
        # Model load may consume RNG; restoration must neutralize it.
        torch.rand(3)
    def finite_forward(self, call):
        return True
    def generate_call(self, call):
        return [{"raw_text": str(torch.rand(()).item()), "fragment_smiles": "C"} for _ in range(4)]


def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU_UNIT_FIXTURE")


def test_real_gpu_required_not_cpu_load_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="ACTUAL_SINGLE_LEASED_GPU"):
        smoke.run_smoke(spec=task(), output_root=tmp_path / "smoke", continue_guard=lambda: True)


def test_two_call_save_fresh_runtime_resume_parity(tmp_path, monkeypatch):
    fake_cuda(monkeypatch)
    result = smoke.run_smoke(spec=task(), output_root=tmp_path / "smoke", continue_guard=lambda: True,
                             runtime_factory=TinyRuntime)
    assert result["resume_parity"] and result["fresh_runtime_instances"] == 3
    assert not result["formal_candidates_adopted"]
    assert not (tmp_path / "smoke/continuous/candidate_pool.jsonl").exists()
    assert smoke.accepted_smoke(tmp_path / "smoke/gpu_smoke.json", task())["compared_attempts"] == 8


def test_smoke_resource_loss_fails_before_model_load(tmp_path, monkeypatch):
    fake_cuda(monkeypatch)
    def forbidden(_):
        raise AssertionError("must not load")
    with pytest.raises(ValueError, match="MAIN_RESOURCE_PRIORITY"):
        smoke.run_smoke(spec=task(), output_root=tmp_path / "smoke", continue_guard=lambda: False,
                        runtime_factory=forbidden)
    assert json.loads((tmp_path / "smoke/gpu_smoke.json").read_text())["state"] == "FAILED"


def test_semantic_comparison_not_pickle_or_object_identity():
    assert smoke.semantic_equal({"x": torch.tensor([1., 2.])}, {"x": torch.tensor([1., 2.])})
    assert not smoke.semantic_equal({"x": torch.tensor([1., 2.])}, {"x": torch.tensor([1., 3.])})


def test_smoke_cannot_adopt_different_variant(tmp_path):
    body = {"state": "GPU_LOAD_GENERATE_CALL_RESUME_PASS", "task_spec_sha256": "wrong",
            "actual_cuda": True, "finite_forward": True, "resume_parity": True}
    atomic_json(tmp_path / "receipt.json", {**body, "self_sha256": canonical_json_sha256(body)})
    with pytest.raises(ValueError, match="NATIVE_GPU_SMOKE_REQUIRED"):
        smoke.accepted_smoke(tmp_path / "receipt.json", task())


def test_verified_loader_does_not_repeat_unchanged_weight_hash(tmp_path, monkeypatch):
    path = tmp_path / "model.bin"
    path.write_bytes(b"tiny-weight-fixture")
    identity = {"path": str(path), "sha256": sha256_file(path)}
    original = runtime.verified_file
    observed = []
    def verify(row):
        observed.append(row)
        return original(row)
    monkeypatch.setattr(runtime, "verified_file", verify)
    runtime.verified_loader_file(identity); runtime.verified_loader_file(identity)
    assert len(observed) == 1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        runtime.verified_loader_file(identity)


def test_terminal_historical_owner_does_not_require_live_heartbeat(tmp_path):
    cfg, _ = helper().source_fixture(tmp_path)
    p = Path(cfg["main_registry_path"]); registry = json.loads(p.read_text())
    current = registry["tasks"][0]
    history = {**current, "task_id": "old", "owner_pid": None, "owner_start_ticks": None,
               "owner_state": "TERMINAL_FAILED_ENGINEERING", "heartbeat": None, "successor_task_id": "t13"}
    updated = build_owner_registry(registry_id="fixture", matrix_authority_root=tmp_path,
        tasks=[history, current], publishers=registry["publishers"], gpu_leases=registry["gpu_leases"], check_processes=False)
    atomic_json(p, updated)
    gpu = GPUObservation(0, "GPU-fixture", "CPU inventory fixture", 1000, 0, 1000, 0)
    result = owner.ResourceSampler(cfg, 0, gpu.uuid, inventory=lambda: [gpu]).sample()
    assert result["source_blockers"] == []
    assert json.loads(p.read_text()) == updated


def test_inode_budget_blocks_without_lowering_guard(tmp_path):
    cfg, _ = helper().source_fixture(tmp_path)
    cfg.update(minimum_free_inodes=2**60, reserved_new_inodes=4096)
    gpu = GPUObservation(0, "GPU-fixture", "CPU inventory fixture", 1000, 0, 1000, 0)
    result = owner.ResourceSampler(cfg, 0, gpu.uuid, inventory=lambda: [gpu]).sample()
    assert not result["storage_safe"] and result["required_free_inodes"] == 2**60 + 4096


def test_evaluation_follows_only_real_completed_pool_and_keeps_registry_separate(tmp_path):
    variant = runtime.VARIANTS[1]
    spec = task(); specfile = tmp_path / "task.json"; atomic_json(specfile, spec)
    row = {"task_spec": {"path": str(specfile), "sha256": sha256_file(specfile)},
           "candidate_root": str(tmp_path / "generated"), "output_root": str(tmp_path / "evaluation"), "command": ["entry"]}
    dispatch = {"variant_order": [variant], "downstream_commands": {variant: row}}
    assert owner.next_completed_evaluation(dispatch) is None
    candidate = tmp_path / "generated"; candidate.mkdir()
    receipt = {"status": "PAUSED_AT_CALL_CHECKPOINT", "spec_sha256": canonical_json_sha256(spec),
               "variant": variant, "next_call": len(spec["calls"])}
    atomic_json(candidate / "candidate_generation_receipt.json", receipt)
    assert owner.next_completed_evaluation(dispatch) is None
    atomic_json(candidate / "candidate_generation_receipt.json", {**receipt, "status": "CANDIDATE_POOL_PASS"})
    assert owner.next_completed_evaluation(dispatch)["variant"] == variant
    evaluation = tmp_path / "evaluation"; evaluation.mkdir()
    atomic_json(evaluation / "final_audit.json", {"state": "PASS", "main_matrix_write": False})
    atomic_json(evaluation / "run_manifest.json", {"task_spec_sha256": sha256_file(specfile)})
    assert owner.next_completed_evaluation(dispatch) is None
    atomic_json(evaluation / "run_manifest.json", {"task_spec_sha256": "wrong"})
    with pytest.raises(ValueError, match="FINAL_BINDING_CHANGED"):
        owner.next_completed_evaluation(dispatch)
