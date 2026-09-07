import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.utils.stage_file_policy import (SCHEMA, canonical_sha, config_file_admission,
    load_stage_policy, stage_file_admission, validate_stage_policy)


def receipt(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def sealed_policy(tmp_path):
    auth = receipt(tmp_path / "authorization.json", dict(allow_stage_based_inode_policy=True,
        inode_base_reserve=20000, inode_next_stage_peak_factor=2, contact_support_first=False))
    rows = [{"component_id": name, "peak_new_files": count, "safe_boundary": "committed_parent",
             "already_existing_files_counted": False, "bound_kind": "CODE_DERIVED"}
            for name, count in (("active", 72), ("new_stage", 128))]
    evidence = receipt(tmp_path / "peak.json", {"state": "CODE_BOUND_FILE_PEAK", "scientific_changes": False,
        "components": {r["component_id"]: {**r, "source_references": ["fixture bounded writer"],
            "derivation": "fixture counts next boundary only"} for r in rows}})
    for row in rows: row["evidence"] = evidence
    body = {"schema_version": SCHEMA, "resource_admission_scope": "NEXT_EXECUTABLE_STAGE",
        "base_reserve": 20000, "peak_factor": 2, "modify_platform_quota": False,
        "apply_to_running_main_science": False, "persistent_root": str(tmp_path),
        "filesystem_device": os.stat(tmp_path).st_dev, "authorization": auth,
        "concurrent_components": rows[:1], "stages": {
            "llm_gpu_generation": {"state": "BOUNDED", "safe_boundary": "committed_parent", "components": rows[1:]},
            "llm_cpu_evaluation": {"state": "PEAK_EVIDENCE_PENDING"}}}
    body["self_sha256"] = canonical_sha(body)
    return body, receipt(tmp_path / "policy.json", body)


def test_new_overlay_applied_without_mutating_original_config(tmp_path):
    policy, identity = sealed_policy(tmp_path)
    cfg = {"persistent_root": str(tmp_path), "minimum_free_inodes": 100000, "reserved_new_inodes": 10761}
    original = copy.deepcopy(cfg)
    assert not config_file_admission(cfg, 94000, stage_id="llm_gpu_generation")["admitted"]
    cfg["stage_file_policy"] = identity
    loaded = load_stage_policy(identity, tmp_path)
    result = config_file_admission(cfg, 94000, stage_id="llm_gpu_generation", policy=loaded)
    assert result["admitted"] and result["required_free_inodes"] == 20400
    assert result["actual_available_file_slots"] == 94000
    assert not result["platform_quota_changed"]
    assert {k: v for k, v in cfg.items() if k != "stage_file_policy"} == original


def test_unknown_future_blocks_only_itself(tmp_path):
    policy, _ = sealed_policy(tmp_path)
    policy["stages"]["llm_cpu_evaluation"]["components"] = [{"peak_new_files": "UNKNOWN"}]
    policy["self_sha256"] = canonical_sha({k: v for k, v in policy.items() if k != "self_sha256"})
    assert stage_file_admission(policy, 94000, stage_id="llm_gpu_generation")["admitted"]
    missing = stage_file_admission(policy, 94000, stage_id="llm_cpu_evaluation")
    assert missing["peak_new_files"] is None and not missing["admitted"]


def test_changed_active_stage_cannot_reuse_old_peak_for_waiting_owner(tmp_path):
    policy, _ = sealed_policy(tmp_path)
    phase = tmp_path / "live-phase.json"
    phase.write_text('{"phase":"TRAINING"}')
    policy["concurrent_components"][0]["scope_guards"] = [{"path": str(phase),
        "allowed_values": {"phase": ["TRAINING"]}}]
    policy["self_sha256"] = canonical_sha({k: v for k, v in policy.items() if k != "self_sha256"})
    assert stage_file_admission(policy, 94000, stage_id="llm_gpu_generation")["admitted"]
    phase.write_text('{"phase":"EVALUATION"}')
    result = stage_file_admission(policy, 94000, stage_id="llm_gpu_generation")
    assert not result["admitted"] and result["pause_requested"]
    assert result["required_free_inodes"] is None


@pytest.mark.parametrize("available,state", [(4999, "RESOURCE_EMERGENCY"),
    (9999, "NEW_SCIENCE_STAGE_FORBIDDEN"), (20199, "CHECKPOINT_AT_SAFE_BOUNDARY"),
    (20399, "WAITING_STAGE_RESOURCE"), (20400, "ADMISSION_PASS")])
def test_runtime_thresholds_use_real_observation(tmp_path, available, state):
    policy, _ = sealed_policy(tmp_path)
    result = stage_file_admission(policy, available, stage_id="llm_gpu_generation", baseline_available=94000)
    assert result["runtime_state"] == state
    assert result["observed_net_slot_consumption"] == 94000 - available


def test_unknown_peak_not_zero_and_hash_change_rejected(tmp_path):
    policy, identity = sealed_policy(tmp_path)
    policy["stages"]["llm_gpu_generation"]["components"][0]["peak_new_files"] = None
    policy["self_sha256"] = canonical_sha({k: v for k, v in policy.items() if k != "self_sha256"})
    with pytest.raises(ValueError, match="UNKNOWN_OR_INVALID"):
        validate_stage_policy(policy)
    Path(identity["path"]).write_text("{}")
    with pytest.raises(ValueError, match="RECEIPT_CHANGED"):
        load_stage_policy(identity, tmp_path)


def test_corrective_cpu_child_and_parent_boundary_use_same_policy():
    root = Path(__file__).resolve().parents[2]
    child = (root / "scripts/ablations/llm/run_bace_common_downstream.py").read_text()
    evaluation = (root / "src/ablations/llm/bace_common_downstream.py").read_text()
    assert 'stage_id="llm_cpu_evaluation"' in child and 'stage_id="llm_cpu_evaluation"' in evaluation
    assert "load_stage_policy" in child and "os.statvfs" in evaluation
    assert '"before_frozen_model_load"' in evaluation
