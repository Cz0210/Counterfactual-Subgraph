"""CPU-only real subprocess/flock tests; never load Torch or request a GPU."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from src.ablations.llm import existing_gpu_owner as owner
from src.ablations.llm import corrected_core_gate as gate
from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils.autodl_runtime import GPUFileLock, ProjectGPUSlotLock, GPUObservation, GPUProcess
from src.utils.final16_owner_registry_v1 import build_owner_registry

ROOT = Path(__file__).resolve().parents[2]


def stat_fixture(root, pid, ticks=123):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    (directory / "stat").write_text(f"{pid} (CPU fixture) S " + " ".join(["0"] * 18 + [str(ticks)]))
    return ticks


def source_fixture(tmp_path, *, reserved=False):
    proc, cgroup = tmp_path / "proc", tmp_path / "memory"
    proc.mkdir(); cgroup.mkdir()
    (proc / "meminfo").write_text("MemAvailable: 1048576 kB\n")
    (cgroup / "memory.max").write_text(str(2 * 1024**3))
    (cgroup / "memory.current").write_text(str(1024**3))
    heartbeat = tmp_path / "main.json"
    atomic_json(heartbeat, {"state": "RUNNING", "pid": 41, "start_ticks": 123, "updated_epoch": time.time()})
    stat_fixture(proc, 41)
    task = {"task_id": "t13", "dataset": "TasteMolNet", "method": "GlobalGCE", "stage": "T13",
            "owner_state": "RUNNING", "owner_pid": 41, "owner_start_ticks": 123, "heartbeat": str(heartbeat),
            "input_root": str(tmp_path / "input"), "output_root": str(tmp_path / "science"),
            "execution_commit": "a" * 40, "task_spec_sha": "b" * 64, "gpu": 1,
            "successor_task_id": None, "publisher_id": "publisher"}
    publisher = {"publisher_id": "publisher", "cell_id": "TasteMolNet/GlobalGCE", "owner_state": "RUNNING",
                 "owner_pid": 41, "owner_start_ticks": 123, "heartbeat": str(heartbeat),
                 "locator": str(tmp_path / "locator"), "lease_path": str(tmp_path / "pub.lease"),
                 "execution_commit": "a" * 40, "claim_enabled": True, "active_writer_count": 0}
    registry = build_owner_registry(registry_id="fixture", matrix_authority_root=tmp_path,
        tasks=[task], publishers=[publisher], gpu_leases=[{"gpu": 1, "task_id": "t13",
            "state": "HELD" if reserved else "RELEASED", "lease_path": str(tmp_path / "gpu1.lease")}], check_processes=False)
    registry_file = tmp_path / "registry.json"
    atomic_json(registry_file, registry)
    cfg = {"main_registry_path": str(registry_file), "main_ready_sources": [str(heartbeat)],
           "proc_root": str(proc), "cgroup_memory_root": str(cgroup), "persistent_root": str(tmp_path),
           "gpu_lock_root": str(tmp_path / "locks"), "minimum_gpu_free_mb": 100,
           "maximum_idle_utilization_percent": 5, "minimum_memory_headroom_bytes": 100,
           "minimum_persistent_free_bytes": 100, "checkpoint_resume_pass": True}
    return cfg, heartbeat


def test_sampler_reopens_sources_and_never_launders_old_timestamp(tmp_path):
    cfg, heartbeat = source_fixture(tmp_path)
    now = [time.time()]
    gpu = GPUObservation(0, "GPU-fixture", "CPU fake GPU inventory", 1000, 0, 1000, 0)
    calls = []
    def inventory():
        calls.append(now[0]); return [gpu]
    sampler = owner.ResourceSampler(cfg, 0, gpu.uuid, inventory=inventory, clock=lambda: now[0], monotonic=lambda: now[0])
    assert sampler.sample()["gpu_idle_seconds"] == 0
    for _ in range(40):
        now[0] += 30
        atomic_json(heartbeat, {"state": "RUNNING", "pid": 41, "start_ticks": 123, "updated_epoch": now[0]})
        result = sampler.sample()
    assert result["gpu_idle_seconds"] == 1200
    assert len(calls) == 41
    old = json.loads(heartbeat.read_text())
    now[0] += 121
    with pytest.raises(ValueError, match="LIVE_SOURCE_STALE"):
        sampler.sample()
    assert json.loads(heartbeat.read_text()) == old
    atomic_json(heartbeat, {**old, "state": "READY_WAITING_GPU", "updated_epoch": now[0]})
    result = sampler.sample()
    assert result["main_ready_waiting_gpu"] and result["gpu_idle_seconds"] == 0


def test_failed_primary_reservation_and_missing_ready_sources_block(tmp_path):
    cfg, heartbeat = source_fixture(tmp_path, reserved=True)
    gpu = GPUObservation(1, "GPU-fixture", "CPU fake", 1000, 0, 1000, 0)
    sampler = owner.ResourceSampler(cfg, 1, gpu.uuid, inventory=lambda: [gpu])
    atomic_json(heartbeat, {"state": "FAILED", "pid": 41, "updated_epoch": time.time()})
    result = sampler.sample()
    assert result["gpu_main_reservation"] and result["gpu_idle_seconds"] == 0
    assert not result["owners_healthy"]
    cfg["main_ready_sources"] = []
    result = sampler.sample()
    assert "MAIN_READY_SOURCE_COVERAGE_UNAVAILABLE" in result["source_blockers"]


def test_busy_inventory_resets_idle_even_with_fresh_json(tmp_path):
    cfg, _ = source_fixture(tmp_path)
    gpu = GPUObservation(0, "GPU-fixture", "CPU fake", 1000, 300, 700, 10,
                         (GPUProcess(99, "main", 300),))
    sampler = owner.ResourceSampler(cfg, 0, gpu.uuid, inventory=lambda: [gpu])
    result = sampler.sample()
    assert result["gpu_idle_seconds"] == 0
    assert "FOREIGN_CUDA_PROCESS_ON_TARGET_GPU" in result["source_blockers"]


class TransportSampler:
    """Fixture ONLY for transport: actual clocks, fake GPU measurement."""
    def __init__(self, config):
        self.config, self.index, self.uuid = config, 0, "GPU-CPU-transport-fixture"
        self.idle_since = None
        self.admitted_idle_seconds = None
    def sample(self, **_):
        return {"schema_version": owner.RESOURCE_SCHEMA, "observed_at": datetime.now(timezone.utc).isoformat(),
                "gpu_index": self.index, "gpu_uuid": self.uuid, "target_gpu_uuid": self.uuid,
                "gpu_lease_mode": "EXCLUSIVE_IDLE", "logical_device": "cuda:0", "gpu_idle_seconds": 1200,
                "gpu_main_reservation": False, "main_ready_waiting_gpu": False,
                "owners_healthy": True, "registry_healthy": True, "memory_safe": True, "storage_safe": True,
                "checkpoint_resume_pass": True, "active_early_ablation_gpus": 0, "borrow_enabled": False}


CHILD = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from src.ablations.llm.existing_gpu_owner import receive_owner_binding, validate_inherited_lease
b = receive_owner_binding()
e = json.loads(Path(b['resource_evidence']).read_text())
assert validate_inherited_lease(e, b['held_gpu_lock_fd'], b['held_project_slot_fd'])
assert os.environ['CUDA_VISIBLE_DEVICES'] == e['gpu_uuid']
assert not os.get_inheritable(b['held_gpu_lock_fd'])
assert not os.get_inheritable(b['held_project_slot_fd'])
forked = os.fork()
if forked == 0:
    for fd in (b['held_gpu_lock_fd'],b['held_project_slot_fd']):
        try: os.fstat(fd)
        except OSError: pass
        else: os._exit(6)
    os._exit(0)
assert os.waitpid(forked,0)[1] == 0
for change in ({'owner_nonce':'wrong'}, {'gpu_child_pid':1}, {'gpu_owner_start_ticks':1},
               {'gpu_lease_mode':'BORROW'}, {'target_gpu_uuid':'GPU-wrong'},
               {'gpu_main_reservation':True}, {'main_ready_waiting_gpu':True},
               {'active_early_ablation_gpus':1}, {'observed_at':'2000-01-01T00:00:00+00:00'}):
    try: validate_inherited_lease(dict(e, **change), b['held_gpu_lock_fd'], b['held_project_slot_fd'])
    except ValueError: pass
    else: raise AssertionError(change)
grandchild = """
import os,sys
from pathlib import Path
from src.utils.autodl_runtime import GPUFileLock,ProjectGPUSlotLock,GPULockError
for descriptor in map(int,sys.argv[1:3]):
    try: os.fstat(descriptor)
    except OSError: pass
    else: raise AssertionError('leaked held FD into grandchild')
for lock in (GPUFileLock(Path(sys.argv[3]),gpu_index=0,gpu_uuid=sys.argv[4]),
             ProjectGPUSlotLock(Path(sys.argv[3])/'llm-ablation',max_slots=1)):
    try: lock.acquire()
    except GPULockError: pass
    else: lock.release(); raise AssertionError('competing process acquired active lease')
"""
subprocess.run([sys.executable,'-c',grandchild,str(b['held_gpu_lock_fd']),str(b['held_project_slot_fd']),
                str(Path(e['gpu_lock_path']).parent),e['gpu_uuid']], close_fds=False,check=True)
Path(sys.argv[1]).write_text(json.dumps({'gate':True,'grandchild_no_fd':True,'competitor_blocked':True}))
if sys.argv[2] == 'pause':
    while True:
        live=json.loads(Path(b['resource_live_evidence']).read_text())
        if live.get('pause_requested'): break
        time.sleep(.01)
    sys.exit(75)
sys.exit(int(sys.argv[2]))
'''


@pytest.mark.parametrize("exit_code", [0, 7, 75])
def test_real_child_fds_competitor_lifecycle_and_grandchild_no_leak(tmp_path, monkeypatch, exit_code):
    cfg, _ = source_fixture(tmp_path)
    # Linux runs real /proc; Mac tests explicitly simulate only start-tick
    # files while all PIDs, subprocesses, inherited FDs and flock are real.
    if Path("/proc/self/stat").exists():
        cfg["proc_root"] = "/proc"
    else:
        monkeypatch.setattr(owner, "process_start_ticks", lambda proc, pid: stat_fixture(Path(proc), pid))
    sampler = TransportSampler(cfg)
    result_path = tmp_path / "result.json"
    environment = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
    run = tmp_path / "owner"
    code = owner.run_owned_child(command=[sys.executable, "-c", CHILD, str(result_path), str(exit_code)],
        environment=environment, sampler=sampler, output_root=run, lock_root=Path(cfg["gpu_lock_root"]),
        run_id="cpu-transport-test", interval=.05)
    assert code == exit_code
    assert json.loads(result_path.read_text())["competitor_blocked"]
    terminal = json.loads((run / "terminal.json").read_text())
    assert terminal["gpu_released_after_child_exit"]
    assert terminal["state"] == {0:"COMPLETE",7:"FAILED",75:"PAUSED"}[exit_code]
    with GPUFileLock(Path(cfg["gpu_lock_root"]), gpu_index=0, gpu_uuid=sampler.uuid):
        with ProjectGPUSlotLock(Path(cfg["gpu_lock_root"]) / "llm-ablation", max_slots=1):
            pass


def test_owner_sigterm_requests_checkpoint_and_waits_for_child(tmp_path, monkeypatch):
    cfg, _ = source_fixture(tmp_path)
    if Path("/proc/self/stat").exists():
        cfg["proc_root"] = "/proc"
    else:
        monkeypatch.setattr(owner, "process_start_ticks", lambda proc, pid: stat_fixture(Path(proc), pid))
    result = tmp_path / "result.json"
    def interrupt():
        for _ in range(500):
            if result.exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(.01)
    thread = threading.Thread(target=interrupt)
    thread.start()
    try:
        code = owner.run_owned_child(command=[sys.executable, "-c", CHILD, str(result), "pause"],
            environment=dict(os.environ, PYTHONPATH=str(ROOT)), sampler=TransportSampler(cfg),
            output_root=tmp_path / "owner", lock_root=Path(cfg["gpu_lock_root"]), run_id="pause", interval=.05)
    finally:
        thread.join(6)
    assert code == 75 and not thread.is_alive()


def test_waiting_owner_does_not_take_any_lease_or_start_process(tmp_path, monkeypatch):
    cfg, _ = source_fixture(tmp_path)
    sampler = TransportSampler(cfg)
    sample = sampler.sample
    sampler.sample = lambda **kw: {**sample(**kw), "gpu_main_reservation": True}
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("must not start a child"))
    assert owner.run_owned_child(command=["not-executed"], environment={}, sampler=sampler,
        output_root=tmp_path / "owner", lock_root=Path(cfg["gpu_lock_root"]), run_id="blocked") == 75
    assert not Path(cfg["gpu_lock_root"]).exists()


def test_cached_corrective_acceptance_never_replays_and_rejects_changed_archive(tmp_path, monkeypatch):
    archive = tmp_path / "corrected.tar.gz"
    archive.write_bytes(b"fixture")
    proof = {"state":"GNN_CORE_SEED7_CORRECTED_PASS","seed":7,
             "validation_counts":{n:187 for n in ("gin","gcn","gatv2","gatedgcn_plus")},
             "counts":{"calibration":288,"test":614},"all_weights_unchanged":True,"gine_unchanged":True,
             "candidate_pool_unchanged":True,"selectors_frozen_before_test":True,"native_common_metrics_replayed":True,
             "raw_ot_recomputed_count":0,"cache_provenance_gaps":[],"main_matrix_write":False,
             "repair_selected_using_test":False,"independent_science_replay_sha256":"a"*64,
             "original_package_sha256":"b"*64,"repair_contract_sha256":"c"*64}
    accepted = tmp_path / "acceptance.json"
    atomic_json(accepted, {"schema_version":"bace_llm_corrected_core_acceptance_v1", "archive_sha256":sha256_file(archive),
        "archive_identity":gate.archive_identity(archive), "independent_audit":proof})
    descriptor = {"path":str(accepted),"sha256":sha256_file(accepted)}
    # No import of temperature-repair verifier is required on this path.
    assert gate.require_corrected_gnn_core(archive,sha256_file(archive),acceptance=descriptor) == proof
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError,match="ACCEPTANCE_CHANGED"):
        gate.require_corrected_gnn_core(archive,"a"*64,acceptance=descriptor)


def test_adopt_existing_sealed_import_receipts_without_replaying_package(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"tiny fixture of already verified transport")
    proof = {"state": "GNN_CORE_SEED7_CORRECTED_PASS", "seed":7,"validation_counts": {n:187 for n in ("gin","gcn","gatv2","gatedgcn_plus")},
             "counts":{"calibration":288,"test":614}, "all_weights_unchanged":True, "gine_unchanged":True,
             "candidate_pool_unchanged":True,"selectors_frozen_before_test":True,"native_common_metrics_replayed":True,
             "raw_ot_recomputed_count":0,"cache_provenance_gaps":[],"main_matrix_write":False,
             "repair_selected_using_test":False,"independent_science_replay_sha256":"a"*64,
             "original_package_sha256":"b"*64,"repair_contract_sha256":"c"*64}
    proof["self_sha256"] = canonical_json_sha256(proof)
    audit = tmp_path / "audit.json"
    atomic_json(audit,proof)
    digest = sha256_file(archive)
    overlay = tmp_path / "overlay.json"
    imported = {**proof,"corrective_audit_sha256":sha256_file(audit), "package_sha256":digest,"sha256":digest,
                "source_archive":str(archive),"bytes":archive.stat().st_size}
    atomic_json(overlay,imported)
    descriptor = gate.adopt_existing_acceptance(archive_path=archive,archive_sha256=digest,
        overlay={"path":str(overlay),"sha256":sha256_file(overlay)},audit={"path":str(audit),"sha256":sha256_file(audit)},
        output_path=tmp_path/"accepted.json")
    assert gate.require_corrected_gnn_core(archive,digest,acceptance=descriptor)==proof
    atomic_json(overlay,{**imported,"corrective_audit_sha256":"0"*64})
    with pytest.raises(ValueError,match="ACCEPTANCE_CHAIN_MISMATCH"):
        gate.adopt_existing_acceptance(archive_path=archive,archive_sha256=digest,
            overlay={"path":str(overlay),"sha256":sha256_file(overlay)},audit={"path":str(audit),"sha256":sha256_file(audit)},
            output_path=tmp_path/"bad.json")


def test_real_cli_can_seal_dispatchable_command_without_gpu_lock(tmp_path, monkeypatch):
    import importlib.util
    loader=importlib.util.spec_from_file_location("successor_owner_test",ROOT/"scripts/ablations/llm/run_bace_llm_successor.py")
    cli=importlib.util.module_from_spec(loader);loader.loader.exec_module(cli)
    cfg,_=source_fixture(tmp_path)
    resource_config=tmp_path/"resources.json";atomic_json(resource_config,cfg)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    readiness={"schema_version":"bace_llm_native_readiness_v1","variants":{}}
    for variant in cli.ORDER:
        task={"variant":variant,"calls":["tinycall"],"generator_state":"LOADER_AND_RESUME_READY_WAITING_GNN_CORE","execution_commit":commit}
        task["task_spec_sha256"]=canonical_json_sha256(task)
        path=tmp_path/(variant+".task.json");atomic_json(path,task)
        readiness["variants"][variant]={"path":str(path),"sha256":sha256_file(path)}
    ready=tmp_path/"readiness.json";atomic_json(ready,readiness)
    archive=tmp_path/"archive";archive.write_bytes(b"accepted fixture")
    monkeypatch.setattr(cli,"require_corrected_gnn_core",lambda *_a,**_kw:{"state":"GNN_CORE_SEED7_CORRECTED_PASS"})
    dispatch=tmp_path/"dispatch.json"
    args=["--readiness",str(ready),"--readiness-sha256",sha256_file(ready),"--output-root",str(tmp_path/"generation"),
          "--gnn-verified-archive",str(archive),"--gnn-verified-archive-sha256",sha256_file(archive),
          "--resource-config",str(resource_config),"--seal-dispatch-spec",str(dispatch)]
    assert cli.main(args)==0
    result=json.loads(dispatch.read_text())
    assert result["state"]=="DISPATCHABLE_WAITING_RESOURCE"
    assert result["command"][1:3]==["-I","-B"] and "--gnn-acceptance" in result["command"]
    assert result["variant_order"]==list(cli.ORDER) and result["borrow_enabled"] is False
    assert not Path(cfg["gpu_lock_root"]).exists()
    first=tmp_path/"generation"/cli.ORDER[0];first.mkdir(parents=True)
    spec=json.loads(Path(readiness["variants"][cli.ORDER[0]]["path"]).read_text())
    paused={"variant":cli.ORDER[0],"status":"PAUSED_AT_CALL_CHECKPOINT","spec_sha256":canonical_json_sha256(spec),"next_call":0}
    atomic_json(first/"candidate_generation_receipt.json",paused)
    atomic_json(first/"latest_checkpoint.json",{"spec_sha256":paused["spec_sha256"],"next_call":0})
    assert cli.next_task(ready,sha256_file(ready),tmp_path/"generation")["resume"]
