import copy
import json

import pytest

from src.utils.t13_performance_dispatch import GIB, SCHEMA, decision, validate_dispatch


def observation():
    return dict(source_blockers=[], memory_safe=False, storage_safe=True,
                memory_headroom_bytes=391*GIB, gpu_main_reservation=False,
                main_ready_waiting_gpu=False, actual_gpu_observation=dict(process_count=0))


def test_live_ram_failure_remains_distinct_from_claim_gap():
    result = decision(dict(canonical_gpu2_diagnostic_claim=None), observation())
    assert result["state"] == "BLOCKED_RESOURCE_AND_CANONICAL_CLAIM"
    assert result["required_headroom_bytes"] == 448*GIB
    assert result["actual_headroom_bytes"] == 391*GIB
    assert result["science_started"] is False
    assert result["gpu_lease_acquired"] is False
    assert "CANONICAL_GPU2_DIAGNOSTIC_CLAIM_NOT_BOUND" in result["blockers"]


def test_free_ram_does_not_fake_a_canonical_claim_or_held_provider():
    live = observation()
    live.update(memory_safe=True, memory_headroom_bytes=460*GIB)
    result = decision(dict(canonical_gpu2_diagnostic_claim=None), live)
    assert result["state"] == "BLOCKED_CANONICAL_CLAIM_AND_HELD_PROVIDER"
    assert result["automatic_waiting_owner_started"] is False
    assert result["registry_modified"] is False


def test_reservation_ready_and_process_are_separate_blockers():
    live = observation()
    live.update(gpu_main_reservation=True, main_ready_waiting_gpu=True,
                actual_gpu_observation=dict(process_count=1))
    result = decision(dict(canonical_gpu2_diagnostic_claim=None), live)
    for blocker in ("MAIN_GPU2_RESERVATION_PRESENT", "MAIN_READY_GPU_TASK_PRESENT", "GPU2_HAS_ACTUAL_PROCESS"):
        assert blocker in result["blockers"]


def test_arbitrary_future_json_cannot_activate_same_lock_bypass():
    live = observation()
    live.update(memory_safe=True)
    result = decision(dict(canonical_gpu2_diagnostic_claim={"status":"PASS"}), live)
    assert "HELD_LEASE_TERMINAL_PROVIDER_ADAPTER_NOT_ACTIVATED" in result["blockers"]
    assert result["science_started"] is False


def test_dispatch_uses_original_provider_without_second_lock_platform():
    from pathlib import Path
    source = (Path(__file__).parents[2] / "src/utils/t13_performance_dispatch.py").read_text()
    assert "from src.ablations.llm.existing_gpu_owner import ResourceSampler" in source
    assert "sampler.sample()" in source
    assert "GPUFileLock(" not in source
    assert "atomic_write_owner_registry" not in source
    assert "subprocess.Popen" not in source


def test_full_run_not_misrepresented_by_parent_receipt():
    result = decision(dict(canonical_gpu2_diagnostic_claim=None), observation())
    assert result["max_full_starts_consumed"] == 0
    assert result["safe_handover_performed"] is False


def test_observation_mode_never_fakes_checkpoint_resume_admission():
    from src.ablations.llm.existing_gpu_owner import validate_resource_config
    cfg = dict(main_registry_path="/registry", main_ready_sources=["/heartbeat"],
        proc_root="/proc", cgroup_memory_root="/cgroup", persistent_root="/runtime",
        gpu_lock_root="/locks", minimum_gpu_free_mb=40000,
        maximum_idle_utilization_percent=5, minimum_memory_headroom_bytes=448*GIB,
        minimum_persistent_free_bytes=100*GIB, checkpoint_resume_pass=False)
    with pytest.raises(ValueError, match="REAL_CHECKPOINT_RESUME"):
        validate_resource_config(cfg)
    assert validate_resource_config(cfg, observation_only=True)["checkpoint_resume_pass"] is False
