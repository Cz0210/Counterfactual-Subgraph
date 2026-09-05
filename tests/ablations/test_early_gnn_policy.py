from src.ablations.gnn.early_policy import hpc_cpu_allowed, gpu_allowed, core_complete


def test_gnn_allowed_at12_on_hpc_cpu():
    assert hpc_cpu_allowed(main_cells=12,bace_reference_pass=True,active_jobs=0)["allowed"]
    assert not hpc_cpu_allowed(main_cells=11,bace_reference_pass=True,active_jobs=0)["allowed"]
    assert not hpc_cpu_allowed(main_cells=12,bace_reference_pass=True,active_jobs=2)["allowed"]


def evidence():
    return dict(main_cells=12,owners_healthy=True,registry_healthy=True,memory_safe=True,storage_safe=True,
                checkpoint_resume_pass=True,main_ready_waiting_gpu=False,gpu_main_reservation=False,
                gpu_idle_seconds=1200,active_early_ablation_gpus=0)


def test_gnn_gpu_reservation_and_idle_gate():
    e=evidence()
    assert gpu_allowed(e,family="gnn")["allowed"]
    for key,value in (("gpu_idle_seconds",1199),("gpu_main_reservation",True),("main_ready_waiting_gpu",True)):
        assert not gpu_allowed({**e,key:value},family="gnn")["allowed"]


def test_llm_gpu_requires_seed7_core_not_secondary_seeds():
    e=evidence()
    assert not gpu_allowed(e,family="llm")["allowed"]
    assert gpu_allowed({**e,"gnn_core_seed7_audit":"PASS"},family="llm")["allowed"]
    assert not core_complete({"status":"PASS","seed":7,"backbones":{"gin":"PASS"}})


def test_gpu_gate_requires_explicit_negative_queue_and_reservations():
    for key in ("main_cells","main_ready_waiting_gpu","gpu_main_reservation","active_early_ablation_gpus"):
        e=evidence()
        e.pop(key)
        assert not gpu_allowed(e,family="gnn")["allowed"]
