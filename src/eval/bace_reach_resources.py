"""Reach-v2 bindings for the existing next-stage resource policy, no new owner."""
import copy
import json
import os
from pathlib import Path

from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, sha256_file
from src.utils.stage_file_policy import canonical_sha, load_stage_policy, config_file_admission


def prepare_resources(prior_path, fresh_root, campaign):
    cfg = read_json(prior_path)
    old = load_stage_policy(cfg["stage_file_policy"], cfg["persistent_root"])
    fresh_root.mkdir(parents=True, exist_ok=False)
    component = "ours_reach_parent_boundary"
    boundary = "one_complete_train_search_parent_or_calibration_parent_atomic_receipt"
    proof = {"state": "CODE_BOUND_FILE_PEAK", "scientific_changes": False,
        "scope": "file-layout admission only; the separately sealed Reach-v2 is an authorized method version",
        "components": {component: {"peak_new_files": 64, "safe_boundary": boundary,
            "source_references": [{"path": str(Path(__file__).parent / "bace_reach_v2.py"),
                                   "sha256": sha256_file(Path(__file__).parent / "bace_reach_v2.py")}],
            "derivation": "At next committed parent: <=12 directory/control entries, one parent JSON plus atomic temp, progress plus temp, <=4 SQLite/compact-node-cache files, <=8 owner heartbeat/evidence/terminal entries, and <=10 freeze/control transition files; 64 conservatively bounds these. Prior committed parent files already exist and are not counted again. No per-match files."}}}
    policy = {k: copy.deepcopy(v) for k, v in old.items() if k != "self_sha256"}
    # The existing T12 owner advanced naturally from reference500 to its
    # already-bound 501..510 reload tail. Same compact writer, still diagnostic.
    # Update only that specific phase, never accept an arbitrary new stage.
    for entry in policy["concurrent_components"]:
        if entry["component_id"] != "t12_next_checkpoint":
            continue
        guard = entry["scope_guards"][0]
        live = read_json(guard["path"])
        if live.get("phase") == "REFERENCE_RESUME_TO_510":
            if (live.get("owner_pid") != 162844 or live.get("owner_start_ticks") != 18577652
                or live.get("reload_tail") != "501-510" or live.get("completed_step") != 500
                or live.get("test_loaded") is not False):
                raise ValueError("T12_RELOAD_TAIL_IDENTITY_NOT_BOUND")
            previous = read_json(entry["evidence"]["path"])["components"]["t12_next_checkpoint"]
            updated = {**previous, "safe_boundary": "reference_reload_tail501_510_next_checkpoint",
                "derivation": previous["derivation"] + " Same 1ad12b56 compact writer now executes existing reload tail501..510, without another reference/full generation; owner identity unchanged."}
            proof["components"]["t12_next_checkpoint"] = updated
            entry["safe_boundary"] = updated["safe_boundary"]
            guard["allowed_values"]["phase"] = ["REFERENCE_RESUME_TO_510"]
            guard["allowed_values"].update(owner_pid=[162844], owner_start_ticks=[18577652], reload_tail=["501-510"])
            guard["maximum_values"] = {"completed_step": 500}
    atomic_json(fresh_root / "file_peak.json", proof)
    evidence = {"path": str(fresh_root / "file_peak.json"), "sha256": sha256_file(fresh_root / "file_peak.json")}
    row = {"component_id": component, "peak_new_files": 64, "safe_boundary": boundary,
           "already_existing_files_counted": False, "bound_kind": "SOURCE_DERIVED_NEXT_BOUNDARY", "evidence": evidence}
    for entry in policy["concurrent_components"]:
        if entry["component_id"] == "t12_next_checkpoint" and "t12_next_checkpoint" in proof["components"]:
            entry["evidence"] = evidence
    # Preserve each still-active main task's code-bound reserve and phase guard.
    # Existing owner method names are protocol slots, not a claim of LLM science.
    policy["stages"] = {name: {"state": "BOUNDED", "safe_boundary": boundary, "components": [row]}
                        for name in ("llm_gpu_generation", "llm_cpu_evaluation")}
    policy["reach_campaign"] = str(campaign)
    policy["self_sha256"] = canonical_sha(policy)
    atomic_json(fresh_root / "policy.json", policy)
    cfg["stage_file_policy"] = {"path": str(fresh_root / "policy.json"), "sha256": sha256_file(fresh_root / "policy.json")}
    atomic_json(fresh_root / "resource_config.json", cfg)
    actual = load_stage_policy(cfg["stage_file_policy"], cfg["persistent_root"])
    result = config_file_admission(cfg, os.statvfs(cfg["persistent_root"]).f_favail,
                                  stage_id="llm_gpu_generation", policy=actual)
    atomic_json(fresh_root / "initial_admission.json", result)
    return {"config": str(fresh_root / "resource_config.json"), "admission": result}


def cpu_boundary(config):
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    disk = os.statvfs(config["persistent_root"])
    result = config_file_admission(config, disk.f_favail, stage_id="llm_cpu_evaluation")
    if (not result["admitted"] or result["pause_requested"]
        or disk.f_bavail * disk.f_frsize < config["minimum_persistent_free_bytes"]
        or memory_headroom(Path(config["proc_root"]), Path(config["cgroup_memory_root"])) < config["minimum_memory_headroom_bytes"]):
        raise SystemExit(75)
