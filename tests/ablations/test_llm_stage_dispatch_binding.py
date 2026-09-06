"""Tiny CPU fixtures: resource-only means no hidden science/command delta."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.ablations.llm import stage_dispatch_binding as binding

DRIVER = "a" * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def signed(value, key="self_sha256"):
    value[key] = binding.canonical_json_sha256({k: v for k, v in value.items() if k != key})
    return value


@pytest.fixture
def case(tmp_path):
    old_root, new_root = tmp_path / "old-driver", tmp_path / "new-driver"
    for root in (old_root, new_root):
        for rel in (binding.GENERATION_ENTRY, binding.DOWNSTREAM_ENTRY):
            p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# tiny entry\n")
    variants = {}
    metadata = {"generator_state": "LOADER_IMPLEMENTED_GPU_SMOKE_REQUIRED_AT_DISPATCH",
                "downstream_state": "EXECUTABLE_ENTRYPOINT_CORRECTED_CORE_CHECK_AT_DISPATCH"}
    for variant in binding.ORDER:
        variants[variant] = {**write_json(tmp_path / (variant + ".json"), signed({
            "variant": variant, "execution_commit": binding.ORIGINAL_SCIENCE_COMMIT,
            "model": {"path": "/DO_NOT_OPEN_MODEL_WEIGHTS", "sha256": "b" * 64},
            "calls": [{"parent_id": "train-1", "attempt": 0}], "seed": 7,
            **metadata}, "task_spec_sha256")), **metadata}
    ready = write_json(tmp_path / "readiness.json", {"schema_version": "bace_llm_native_readiness_v1", "variants": variants})
    resource = {"minimum_free_inodes": 100000, "minimum_memory_headroom_bytes": 64 * 1024**3,
                "minimum_persistent_free_bytes": 100 * 1024**3, "gpu_lock_root": "/existing/locks",
                "main_registry_path": "/existing/registry/current.json"}
    old_resource = write_json(tmp_path / "old-resource.json", resource)
    old = {"schema_version": binding.DISPATCH_SCHEMA, "execution_commit": binding.ORIGINAL_SCIENCE_COMMIT,
           "command": ["/python", "-I", "-B", str(old_root / binding.GENERATION_ENTRY),
                       "--readiness", ready["path"], "--output-root", str(tmp_path / "generation")],
           "readiness": ready, "resource_config": old_resource, "max_llm_gpus": 1,
           "borrow_enabled": False, "variant_order": binding.ORDER, "science_started": False,
           "state": "DISPATCHABLE_WAITING_RESOURCE", "main_matrix_count_required": False,
           "secondary_seeds_required": False, "evaluation_policy": {"rule_budget": "AT_MOST_K", "max_rules": 20},
           "downstream_commands": {}}
    for variant in binding.ORDER:
        old["downstream_commands"][variant] = {"candidate_root": str(tmp_path / "generation" / variant),
            "output_root": str(tmp_path / "evaluation" / variant), "task_spec": variants[variant],
            "command": ["/python", "-I", "-B", str(old_root / binding.DOWNSTREAM_ENTRY),
                "--task-spec", variants[variant]["path"], "--output-root", str(tmp_path / "evaluation" / variant),
                "--device", "cpu", "--cpu-threads", "2"]}
    original = write_json(tmp_path / "original-dispatch.json", signed(old))
    policy = write_json(tmp_path / "stage-policy.json", {"schema_version": "tiny_stage_file_policy", "guard_lowered": False})
    new_resource = write_json(tmp_path / "new-resource.json", {**resource, "stage_file_policy": policy})
    output = tmp_path / "fresh-dispatch.json"
    return dict(original_dispatch=original, resource_config=new_resource,
                owner_driver_commit=DRIVER, project_root=new_root, output_path=output,
                old=old, old_root=old_root, policy=policy)


def seal(case):
    return binding.seal_resource_dispatch(**{k: case[k] for k in (
        "original_dispatch", "resource_config", "owner_driver_commit", "project_root", "output_path")})


def read_sealed(case):
    return json.loads(case["output_path"].read_text())


def test_seal_preserves_original_generation_tasks_models_and_outputs(case):
    before = Path(case["original_dispatch"]["path"]).read_bytes()
    result = seal(case); new = read_sealed(case)
    assert result["state"] == "SEALED_WAITING_RESOURCE"
    assert not result["science_started"] and not result["owner_started"]
    assert new["command"] == case["old"]["command"]
    assert new["readiness"] == case["old"]["readiness"]
    assert Path(case["original_dispatch"]["path"]).read_bytes() == before
    for variant in binding.ORDER:
        row = new["downstream_commands"][variant]
        old = case["old"]["downstream_commands"][variant]
        assert {k: v for k, v in row.items() if k != "command"} == {k: v for k, v in old.items() if k != "command"}
        assert row["command"][:3] == old["command"][:3]
        assert row["command"][4:-5] == old["command"][4:]
        assert row["command"][-5:] == ["--stage-file-policy", case["policy"]["path"],
            "--stage-file-policy-sha256", case["policy"]["sha256"], "--compact-node-cache"]
    result = binding.validate_dispatch_runtime(new, DRIVER, case["project_root"])
    assert result["science_execution_commit"] == binding.ORIGINAL_SCIENCE_COMMIT
    assert result["owner_driver_commit"] == DRIVER
    assert not result["resource_admission_evaluated"]


@pytest.mark.parametrize("mutation", ["unknown_field", "missing_field", "metadata_drift"])
def test_actual_readiness_reference_metadata_is_checked_not_silently_discarded(case, mutation):
    original = deepcopy(case["old"])
    path = Path(original["readiness"]["path"])
    readiness = json.loads(path.read_text())
    ref = readiness["variants"][binding.ORDER[0]]
    if mutation == "unknown_field": ref["command"] = ["/unapproved/command"]
    elif mutation == "missing_field": del ref["generator_state"]
    else: ref["generator_state"] = "GPU_SMOKE_ALREADY_PASS"
    original["downstream_commands"][binding.ORDER[0]]["task_spec"] = deepcopy(ref)
    original["readiness"] = write_json(path, readiness)
    case["original_dispatch"] = write_json(Path(case["original_dispatch"]["path"]), signed(original))
    with pytest.raises(ValueError, match="TASK_REFERENCE_"): seal(case)
    assert not case["output_path"].exists()


@pytest.mark.parametrize("field,value", [
    ("command", ["/different", "-I", "-B", "/different.py"]),
    ("readiness", {"path": "/different.json", "sha256": "b" * 64}),
    ("science_started", True), ("borrow_enabled", True), ("max_llm_gpus", 2),
    ("max_llm_gpus", True), ("main_matrix_count_required", True),
    ("secondary_seeds_required", True), ("state", "RUNNING"),
    ("evaluation_policy", {"rule_budget": "PAD_TO_K", "max_rules": 20}),
    ("new_science_field", "not allowed"), ("variant_order", list(reversed(binding.ORDER))),
])
def test_any_non_resource_dispatch_change_rejected_even_with_new_self_hash(case, field, value):
    seal(case); new = read_sealed(case); new[field] = value; signed(new)
    with pytest.raises(ValueError):
        binding.validate_dispatch_runtime(new, DRIVER, case["project_root"])


@pytest.mark.parametrize("mutation", ["output", "task", "cpu_threads", "extra_flag", "no_compact", "wrong_script"])
def test_downstream_only_exact_entry_and_resource_tail_may_change(case, mutation):
    seal(case); new = read_sealed(case); row = new["downstream_commands"][binding.ORDER[0]]
    if mutation == "output": row["output_root"] += "-changed"
    elif mutation == "task": row["task_spec"]["sha256"] = "f" * 64
    elif mutation == "cpu_threads": row["command"][row["command"].index("--cpu-threads") + 1] = "8"
    elif mutation == "extra_flag": row["command"].append("--skip-test-gate")
    elif mutation == "no_compact": row["command"].pop()
    else: row["command"][3] = "/different/run_bace_common_downstream.py"
    signed(new)
    with pytest.raises(ValueError): binding.validate_dispatch_runtime(new, DRIVER, case["project_root"])


def test_resource_threshold_or_registry_cannot_change(case):
    resource = json.loads(Path(case["resource_config"]["path"]).read_text())
    resource["minimum_free_inodes"] = 1
    case["resource_config"] = write_json(Path(case["resource_config"]["path"]), resource)
    with pytest.raises(ValueError, match="NON_POLICY_DRIFT"): seal(case)
    assert not case["output_path"].exists()


def test_changed_small_policy_or_original_dispatch_rejected(case):
    seal(case); new = read_sealed(case)
    Path(case["policy"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="FILE_SHA_MISMATCH"):
        binding.validate_dispatch_runtime(new, DRIVER, case["project_root"])


def test_original_missing_source_spec_rejected_without_model_access(case):
    Path(case["old"]["downstream_commands"][binding.ORDER[0]]["task_spec"]["path"]).unlink()
    with pytest.raises(ValueError, match="PHYSICAL_SMALL_JSON"): seal(case)


def test_wrong_driver_and_wrong_root_rejected(case):
    seal(case); new = read_sealed(case)
    with pytest.raises(ValueError, match="OWNER_DRIVER_COMMIT"):
        binding.validate_dispatch_runtime(new, "f" * 40, case["project_root"])
    with pytest.raises(ValueError, match="OWNER_SCIENCE_ROLE_BINDING"):
        binding.validate_dispatch_runtime(new, DRIVER, case["old_root"])


def test_no_overwrite_or_nested_overlay(case):
    seal(case); before = case["output_path"].read_bytes()
    with pytest.raises(ValueError, match="FRESH_DISPATCH_DESTINATION"): seal(case)
    assert case["output_path"].read_bytes() == before
    case["original_dispatch"] = binding.small_json_descriptor(case["output_path"])
    with pytest.raises(ValueError, match="OVERLAY_CHAIN"): seal(case)


def test_legacy_same_driver_semantics_remain_supported(case):
    result = binding.validate_dispatch_runtime(case["old"], binding.ORIGINAL_SCIENCE_COMMIT, case["old_root"])
    assert result["resource_only_overlay"] is False
    with pytest.raises(ValueError): binding.validate_dispatch_runtime(case["old"], DRIVER, case["project_root"])


def test_cli_seal_is_cpu_only_and_real_entrypoint(case, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts/autodl/rebind_llm_stage_resource_policy.py"
    spec = importlib.util.spec_from_file_location("stage_rebind_cli", path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", case["project_root"])
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: DRIVER)
    assert cli.main(["--config", "configs/hpc.yaml", "--action", "seal",
        "--original-dispatch", case["original_dispatch"]["path"], "--original-dispatch-sha256", case["original_dispatch"]["sha256"],
        "--resource-config", case["resource_config"]["path"], "--resource-config-sha256", case["resource_config"]["sha256"],
        "--output", str(case["output_path"])]) == 0
    assert json.loads(capsys.readouterr().out)["owner_started"] is False


def test_oversized_or_symlink_json_is_not_reopened(tmp_path):
    path = tmp_path / "large.json"; path.write_bytes(b" " * (binding.MAX_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="SIZE_LIMIT"): binding.small_json_descriptor(path)
    link = tmp_path / "link.json"; link.symlink_to(path)
    with pytest.raises(ValueError, match="PHYSICAL_SMALL_JSON"): binding.small_json_descriptor(link)
