"""Read-only, exact startup-failure supersession for the nominated Mut repair.

This proves only that the preserved pre-science failure was followed by the
three original successful stages. Ordinary scientific terminal validation is
still mandatory at every caller. No payload, model or cache is opened here.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256
from src.eval.am_legacy_standardization import scan_live_writers

SCHEMA = "mut_pre_science_startup_failure_supersession_v1"
REPAIR_COMMIT = "94300ea34724088f3336fab95aba5ea5855f58e3"
REPAIR_DRIVER_SHA256 = "2818b8bb742c3d8d2d9e16306ca72e9b96bac9a8381355202ec25f23334d8939"
SCIENCE_COMMIT = "fbefa4caff172453d42afd90f8518bc7e8bddf47"
FAILURE_MESSAGE = "CONTINUATION_PREVIOUS_STAGE_STATE_INVALID:unified_eval"
STAGES = ("unified_eval", "full_gate", "freeze")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError("MUT_STARTUP_SUPERSESSION_INVALID:" + label)


def _time(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value))
    _require(result.tzinfo is not None, "timestamp_not_aware")
    return result


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def validate_startup_repair_supersession(
    root: Path, receipt_path: Path, *, proc_root: Path
) -> dict[str, Any]:
    """Accept only the pinned control-only repair; preserve every old byte."""
    inventory: dict[str, dict[str, Any]] = {}

    def file(path: Path) -> Path:
        _require(path.is_absolute() and path.resolve(strict=True) == path
                 and path.is_file() and path.stat().st_size <= 8 * 1024 * 1024,
                 "small_physical_file:" + str(path))
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(all(getattr(before, k) == getattr(after, k) for k in fields), "file_changed:" + str(path))
        inventory[str(path)] = {"bytes": before.st_size, "sha256": digest}
        return path

    def read(path: Path) -> dict[str, Any]:
        value = json.loads(file(path).read_text())
        _require(isinstance(value, dict), "object:" + str(path))
        return value

    def self_hash(value: dict[str, Any]) -> bool:
        return value.get("self_sha256") == stable_json_sha256(
            {k: v for k, v in value.items() if k != "self_sha256"})

    _require(root.is_absolute() and root.resolve(strict=True) == root, "science_root")
    receipt = read(receipt_path)
    control = receipt_path.parent
    _require(receipt_path.name == "control_adapter_receipt.json"
             and not control.is_relative_to(root) and not root.is_relative_to(control), "control_root")
    expected = dict(schema_version="mut_chemistry_startup_control_repair_v1",
        state="CONTROL_ADAPTER_READY_NOT_SCIENCE_PASS", output_root=str(root),
        control_output_root=str(control), control_driver_commit=REPAIR_COMMIT,
        control_driver_sha256=REPAIR_DRIVER_SHA256, scientific_project_commit=SCIENCE_COMMIT,
        scientific_argv_changed=False, scientific_import_tree_changed=False,
        source_generation_rerun=False, common_recourse_rerun=False, chemistry_rerun=False,
        original_stage_state_rewritten=False, original_failure_rewritten=False,
        control_changes=["_run_stage.output_root", "_run_stage.checkpoint_path",
                         "runner.write_json:exact_prior_FAILED.json_only"])
    _require(self_hash(receipt) and all(receipt.get(k) == v for k, v in expected.items()), "repair_receipt")
    driver = file(Path(receipt["control_driver_path"]))
    _require(inventory[str(driver)]["sha256"] == REPAIR_DRIVER_SHA256, "repair_driver_bytes")
    source = Path(receipt["scientific_source_root"])
    _require(source.is_absolute() and source.resolve(strict=True) == source, "science_source")
    _require(subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True, timeout=30).strip() == SCIENCE_COMMIT, "science_source_commit")
    _require(not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
        "--untracked-files=all", "--", "scripts", "src", "configs"],
        text=True, timeout=30).strip(), "science_source_dirty")
    for parent in (root, control, root / "unified_eval", root / "full_gate", root / "standardized"):
        for name in ("FAILED", "FAILED.json", "FAIL.json", "runner_FAILED.json"):
            if parent == root and name == "FAILED.json":
                continue
            _require(not _present(parent / name), "additional_failure:" + str(parent / name))
    names = ("FAILED.json", "stage_state.json", "chemistry_stage_boundary.json")
    _require(set(receipt.get("preserved", {})) == set(names), "preserved_inventory")
    originals = {}
    for name in names:
        originals[name] = read(root / name)
        file(control / "preserved" / name)
        _require(inventory[str(root / name)]["sha256"] == receipt["preserved"][name]
                 == inventory[str(control / "preserved" / name)]["sha256"], "preserved_bytes:" + name)
    failure, old_stage, boundary = (originals[name] for name in names)
    _require(set(failure) == {"schema_version", "status", "dataset", "error_class", "message", "output_root", "failed_at"}
        and failure.get("schema_version") == "mut_comrecgc_fast_accurate_standardization_failure_v2"
        and failure.get("status") == "FAILED" and failure.get("dataset") == "mutagenicity"
        and failure.get("error_class") == "ValueError" and failure.get("message") == FAILURE_MESSAGE
        and failure.get("output_root") == str(root), "exact_historical_failure")
    _require(self_hash(boundary) and boundary.get("schema_version") == "mut_train_only_chemistry_stage_boundary_v1"
        and boundary.get("state") == "SEALED_CHEMISTRY_WAITING_EVALUATION_ADMISSION"
        and boundary.get("chemistry_complete") is True and boundary.get("test_evaluation_started") is False,
        "sealed_chemistry_boundary")
    closure_names = ("chemistry/_RUN_COMPLETE.json", "chemistry/run_manifest.json",
        "chemistry/final_artifact_audit.json", "generation_adoption_manifest.json",
        "historical_adoption_manifest.json", "upstream_checkout_audit.json")
    sealed = boundary.get("sealed_files", [])
    _require({row["path"] for row in sealed} == {str(root / name) for name in closure_names}
             and len(sealed) == len(closure_names), "chemistry_closure_inventory")
    for row in sealed:
        path = file(Path(row["path"]))
        _require(inventory[str(path)]["sha256"] == row["sha256"], "chemistry_closure_bytes")
    contract = boundary["contract"]
    commands = contract["commands"]
    _require(contract.get("project_commit") == SCIENCE_COMMIT
             and [row[0] for row in commands] == ["chemistry", *STAGES], "science_commands")
    _require(old_stage.get("schema_version") == 2 and old_stage.get("status") == "PASS"
        and old_stage.get("stage") == "chemistry"
        and old_stage.get("argv_sha256") == stable_json_sha256(commands[0][1]), "prior_chemistry_pass")
    previous = _time(failure["failed_at"])
    _require(_time(old_stage["completed_at"]) <= _time(boundary["completed_at"]) < previous,
             "failure_not_after_sealed_chemistry")
    spec_path = Path(receipt["continuation_spec_path"])
    spec = read(spec_path)
    _require(inventory[str(spec_path)]["sha256"] == receipt["continuation_spec_sha256"]
        and self_hash(spec) and spec["self_sha256"] == receipt["continuation_spec_self_sha256"]
        and spec.get("execution_commit") == SCIENCE_COMMIT and spec.get("cwd") == str(source)
        and spec.get("output_root") == str(root)
        and spec.get("argv_sha256") == receipt["original_argv_sha256"] == stable_json_sha256(spec["command"])
        and "--resume-after-chemistry" in spec["command"], "sealed_continuation")
    checkpoints = {stage: str(control / "stage_checkpoints" / (stage + ".json")) for stage in STAGES}
    _require(receipt.get("stage_checkpoint_paths") == checkpoints, "checkpoint_paths")
    marker_contracts = (("unified_eval/_RUN_COMPLETE.json", "run_complete", "run_slot_unified_eval.py"),
        ("full_gate/gate_result.json", "audit_passed", "gate_recovery.py"),
        ("standardized/_FINALIZED.json", "finalized", "freeze_recovery_result.py"))
    owners = set()
    for command, (relative_marker, field, script) in zip(commands[1:], marker_contracts):
        stage, argv, marker, required = command
        _require(marker == str(root / relative_marker) and required == field
            and argv[1] == str(source / "scripts/baselines/comrecgc" / script), "stage_command:" + stage)
        checkpoint = read(Path(checkpoints[stage]))
        digest = stable_json_sha256(argv)
        _require(checkpoint.get("schema_version") == 2 and checkpoint.get("status") == "PASS"
            and checkpoint.get("stage") == stage and checkpoint.get("argv_sha256") == digest
            and checkpoint.get("marker") == marker and checkpoint.get("required_field") == field
            and checkpoint.get("process_group_contract") == "durable_barrier_dedicated_child_session_v2",
            "stage_checkpoint:" + stage)
        _require(previous < _time(checkpoint["started_at"]) <= _time(checkpoint["child_started_at"])
            <= _time(checkpoint["completed_at"]), "stage_order:" + stage)
        previous = _time(checkpoint["completed_at"])
        marker_value = read(Path(marker))
        _require(marker_value.get(field) is True and checkpoint.get("marker_sha256")
                 == inventory[marker]["sha256"], "stage_marker:" + stage)
        binding = checkpoint["startup_barrier"]
        record_path = control / "stage_checkpoints" / ("." + stage + ".exec-startup.00.json")
        lock_path = control / "stage_checkpoints" / ("." + stage + ".exec-startup.lock")
        _require(binding.get("schema_version") == "comrecgc_continuation_exec_startup_binding_v1"
            and binding.get("phase") == "BOUND" and binding.get("generation") == 0
            and binding.get("stage") == stage and binding.get("target_argv_sha256") == digest
            and binding.get("record_path") == str(record_path)
            and binding.get("lock_path") == str(lock_path), "stage_barrier:" + stage)
        record = read(record_path)
        _require(inventory[str(record_path)]["sha256"] == binding.get("record_sha256")
            and record.get("schema") == "autodl_exec_startup_barrier_v1"
            and record.get("target_argv") == argv and record.get("target_argv_sha256") == digest
            and record.get("record_path") == str(record_path) and record.get("lock_path") == str(lock_path)
            and stable_json_sha256(record["launcher_argv"]) == binding.get("launcher_argv_sha256")
            == record.get("launcher_argv_sha256"), "stage_barrier_argv:" + stage)
        owner, child = int(checkpoint["runner_pid"]), int(checkpoint["child_pid"])
        _require(owner > 1 and child > 1 and checkpoint.get("process_group_id") == child
            and int(checkpoint["child_start_ticks"]) > 0, "stage_owner_identity:" + stage)
        owners.add(owner)
        for pid in (owner, child):
            _require(not (proc_root / str(pid)).exists(), "stage_process_still_present:" + str(pid))
    _require(len(owners) == 1 and read(control / "stage_state.json") == checkpoint, "final_control_stage")
    final = read(root / "final_gate.json")
    _require(final.get("status") == "PASS" and final.get("project_commit") == SCIENCE_COMMIT
        and final.get("independent_scientific_adoption_authorized") is True
        and _time(final["completed_at"]) >= previous
        and read(root / "run_manifest.json") == final
        and read(root / "_RUN_COMPLETE.json") == {**final, "run_complete": True}, "later_final_closure")
    audits = {str(p): scan_live_writers(p, proc_root=proc_root) for p in (root, control)}
    _require(all(a.get("procfs_verified") is True and a.get("writable_fd_count") == 0
                 and a.get("writers") == [] for a in audits.values()), "live_writer")
    result = dict(schema_version=SCHEMA, status="RESOLVED_PRE_SCIENCE_STARTUP_FAILURE",
        repair_receipt_path=str(receipt_path), repair_receipt_sha256=inventory[str(receipt_path)]["sha256"],
        repair_receipt_self_sha256=receipt["self_sha256"], repair_control_commit=REPAIR_COMMIT,
        scientific_project_commit=SCIENCE_COMMIT, historical_failure_preserved=True,
        scientific_validation_bypassed=False, three_original_stages_completed=True,
        final_gate_sha256=inventory[str(root / "final_gate.json")]["sha256"],
        writer_audits=audits, inventory=inventory)
    result["self_sha256"] = stable_json_sha256(result)
    return result
