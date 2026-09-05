import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
import uuid

import pytest

from src.ablations.llm import corrected_core_gate as gate
from src.ablations.llm import bace_l0_hpc as l0
from src.ablations.llm.portable_inputs import PortableInputs, SCHEMA
from src.ablations.llm.contracts import canonical_json_sha256
from src.ablations.gnn.early_policy import gpu_allowed
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils import gnn_seed7_mac_relay as relay

ROOT = Path(__file__).resolve().parents[1]


def load_cli(relative):
    spec = importlib.util.spec_from_file_location("corrected_successor_test_cli", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_gate_calls_independent_verifier_readonly(tmp_path, monkeypatch):
    archive = tmp_path / "corrected.tar.gz"
    archive.write_bytes(b"tiny-byte-fixture")
    observed = []
    verifier = ModuleType("src.ablations.gnn.temperature_repair")
    def verify(package, output_root=None):
        observed.append((package, output_root))
        return {"state": "GNN_CORE_SEED7_CORRECTED_PASS", "validation_counts": {"gin": 187}}
    verifier.verify_corrective_package = verify
    monkeypatch.setitem(sys.modules, verifier.__name__, verifier)
    result = gate.require_corrected_gnn_core(archive, sha256_file(archive))
    assert observed == [(archive, None)]
    assert result["main_matrix_count_required"] is False
    assert result["secondary_seeds_required"] is False
    assert not result["gpu_borrow_enabled"]
    verifier.verify_corrective_package = lambda *_a, **_k: {"state": "PASS"}
    with pytest.raises(ValueError, match="CORRECTED_PASS"):
        gate.require_corrected_gnn_core(archive, sha256_file(archive))


def test_corrected_gate_rejects_old_not_fit_and_corrupt_bytes(tmp_path, monkeypatch):
    archive = tmp_path / "old.tar.gz"
    archive.write_bytes(b"old")
    verifier = ModuleType("src.ablations.gnn.temperature_repair")
    def reject(*_a, **_kw): raise ValueError("old_not_fit")
    verifier.verify_corrective_package = reject
    monkeypatch.setitem(sys.modules, verifier.__name__, verifier)
    with pytest.raises(ValueError, match="old_not_fit"):
        gate.require_corrected_gnn_core(archive, sha256_file(archive))
    with pytest.raises(ValueError, match="SHA"):
        gate.require_corrected_gnn_core(archive, "0" * 64)


def portable_fixture(tmp_path):
    root = tmp_path / "portable"
    root.mkdir()
    task = root / "task.json"
    atomic_json(task, {"variant": "BRICS_FIXED"})
    identity = {"path": "/autodl-fs/data/original/task.json", "sha256": sha256_file(task)}
    manifest = {"schema_version": SCHEMA, "variant": "BRICS_FIXED", "original_manifests_modified": False,
                "model_weights_copied": False, "task_spec": identity, "brics_root_relative": "brics",
                "source_files": {identity["path"]: {"relative": "task.json", "sha256": identity["sha256"], "size": task.stat().st_size}}}
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(root / "portable_manifest.json", manifest)
    return root, identity, manifest


def test_portable_mapping_retains_original_json_and_rejects_wrong_hash(tmp_path):
    root, identity, _ = portable_fixture(tmp_path)
    portable = PortableInputs(root)
    before = (root / "task.json").read_bytes()
    assert portable.resolve(identity) == root / "task.json"
    assert portable.task_spec_path().read_bytes() == before
    with pytest.raises(ValueError, match="BINDING"):
        portable.resolve({**identity, "sha256": "b" * 64})
    (root / "task.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="BYTES_CHANGED"):
        portable.resolve(identity)


@pytest.mark.parametrize("relative", ["../outside.json", "/autodl-fs/data/other.json"])
def test_portable_mapping_rejects_path_escape(tmp_path, relative):
    root, identity, manifest = portable_fixture(tmp_path)
    manifest["source_files"][identity["path"]]["relative"] = relative
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(root / "portable_manifest.json", manifest)
    with pytest.raises(ValueError, match="PATH_ESCAPE"):
        PortableInputs(root).resolve(identity)


def test_cpu_l0_executes_common_route_without_matrix_or_gpu_gate(tmp_path, monkeypatch):
    from src.ablations.llm import bace_common_downstream
    root, _, _ = portable_fixture(tmp_path)
    observed = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bace_common_downstream, "run_downstream", lambda **kw: observed.append(kw) or {"state": "PASS"})
    result = l0.run_l0(portable_input_bundle=root, gnn_input_bundle=tmp_path / "gnn",
        corrected_gnn_archive=tmp_path / "corrected", corrected_gnn_sha256="a" * 64,
        registry_root=tmp_path / "registry", output_root=tmp_path / "run")
    assert result["state"] == "PASS"
    assert observed[0]["device"] == "cpu" and observed[0]["portable_input_bundle"] == root
    assert "main_cells" not in observed[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-main")
    with pytest.raises(ValueError, match="NO_VISIBLE_GPU"):
        l0.run_l0(portable_input_bundle=root, gnn_input_bundle="unused", corrected_gnn_archive="unused",
            corrected_gnn_sha256="a" * 64, registry_root="unused", output_root="unused")


def test_gpu_priority_blocks_reserved_idle_card_without_matrix_count():
    base = {"main_cells": 0, "owners_healthy": True, "registry_healthy": True, "memory_safe": True,
            "storage_safe": True, "checkpoint_resume_pass": True, "main_ready_waiting_gpu": False,
            "gpu_main_reservation": False, "gpu_idle_seconds": 1200, "active_early_ablation_gpus": 0,
            "gnn_core_seed7_audit": "PASS"}
    assert gpu_allowed(base, family="llm")["allowed"]
    for changes in ({"gpu_main_reservation": True}, {"main_ready_waiting_gpu": True},
                    {"gpu_idle_seconds": 1199}, {"active_early_ablation_gpus": 1}):
        assert not gpu_allowed({**base, **changes}, family="llm")["allowed"]


def successor_fixture(tmp_path):
    cli = load_cli("scripts/ablations/llm/run_bace_llm_successor.py")
    readiness = {"schema_version": "bace_llm_native_readiness_v1", "variants": {}}
    for variant in cli.ORDER:
        spec = {"variant": variant, "calls": ["call"], "generator_state": "LOADER_AND_RESUME_READY_WAITING_GNN_CORE"}
        spec["task_spec_sha256"] = canonical_json_sha256(spec)
        path = tmp_path / (variant + ".json")
        atomic_json(path, spec)
        readiness["variants"][variant] = {"path": str(path), "sha256": sha256_file(path)}
    path = tmp_path / "readiness.json"
    atomic_json(path, readiness)
    return cli, path


def test_one_shot_generation_order_and_resume_without_lease_creation(tmp_path):
    cli, readiness = successor_fixture(tmp_path)
    root = tmp_path / "outputs"
    next_result = cli.next_task(readiness, sha256_file(readiness), root)
    assert next_result["variant"] == "CHEMLLM_7B_OFF_THE_SHELF"
    first = root / cli.ORDER[0]
    first.mkdir(parents=True)
    atomic_json(first / "latest_checkpoint.json", {"preserved": True})
    assert cli.next_task(readiness, sha256_file(readiness), root)["resume"]
    spec = json.loads(Path(next_result["task_spec"]).read_text())
    (first / "candidate_pool.jsonl").write_text("{}\n")
    atomic_json(first / "candidate_generation_receipt.json", {"status": "CANDIDATE_POOL_PASS",
        "variant": cli.ORDER[0], "spec_sha256": canonical_json_sha256(spec), "next_call": 1,
        "candidate_pool_sha256": sha256_file(first / "candidate_pool.jsonl")})
    assert cli.next_task(readiness, sha256_file(readiness), root)["variant"] == "CHEMLLM_7B_PPO_LORA_MAIN"
    assert not (root / "gpu.lock").exists()


def test_corrective_relay_is_scoped_and_cannot_adopt_old_receipt(tmp_path):
    plan = relay.CorrectiveRelayPlan(str(uuid.uuid4()), ("123", "124"), "a" * 40,
                                    Path("/root/autodl-tmp/worktrees/corrected-a"))
    assert plan.hpc_receipt == relay.CORRECTED_HPC_ROOT / "result_package.json"
    assert plan.import_root.parent == relay.CORRECTED_AUTODL_PARENT
    receipt = {"state": "GNN_CORE_SEED7_CORRECTED_PASS", "main_matrix_write": False,
        "path": str(relay.CORRECTED_HPC_ROOT / relay.CORRECTED_ARCHIVE_NAME), "bytes": 1, "sha256": "b" * 64,
        "scientific_engine_commit": relay.SCIENTIFIC_ENGINE_COMMIT, "repair_driver_commit": "a" * 40}
    assert relay.validate_receipt(receipt, plan) == receipt
    with pytest.raises(relay.RelayError, match="fixed campaign"):
        relay.validate_receipt({**receipt, "path": str(relay.HPC_ROOT / relay.ARCHIVE_NAME)}, plan)
    command = " ".join(relay.import_command(plan, receipt))
    assert "import_corrected_gnn_core.py" in command and "CUDA_VISIBLE_DEVICES=" in command
    assert "run_bace_native_llm" not in command and "fast16_matrix_authority" not in command


def test_cpu_pair_has_intel_no_gres_and_afterok_contract():
    text = (ROOT / "scripts/slurm/run_bace_l0_cpu.sh").read_text()
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert "--dependency=afterok:" in text
    assert '--corrected-package-receipt "$GNN_CORRECTED_PACKAGE_RECEIPT"' in text
    assert '--device cuda' not in text
