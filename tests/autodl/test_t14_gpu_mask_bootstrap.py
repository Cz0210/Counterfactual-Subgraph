"""Exercise the real generic launch producer and strict T14 GPU consumer."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

from scripts.autodl import gpu_lock
from src.utils.tastemolnet_t9_managed_v2 import require_gpu_runtime


def test_generic_gpu_lock_environment_satisfies_t14_consumer(monkeypatch,tmp_path):
    args=SimpleNamespace(action='run',llm_dispatch_spec=None,command=['--','python','science'],
        project_root=tmp_path,data_root=tmp_path,run_id='tiny-t14',gpu_index=2,gpu_uuid='GPU-exact-test')
    monkeypatch.setattr(gpu_lock,'parser',lambda:SimpleNamespace(parse_args=lambda:args))
    monkeypatch.setattr(gpu_lock,'resolve_project_root',lambda value:value)
    monkeypatch.setattr(gpu_lock,'select_data_root',lambda root,explicit:explicit)
    layout=SimpleNamespace(locks_dir=tmp_path);layout.ensure=lambda:layout
    monkeypatch.setattr(gpu_lock,'build_runtime_layout',lambda **kwargs:layout)
    monkeypatch.setattr(gpu_lock,'GPUFileLock',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(gpu_lock,'sanitized_environment',lambda:{})
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1)))
    observed=[]
    def child(command,*,env,check):
        for key,value in env.items():monkeypatch.setenv(key,value)
        require_gpu_runtime('GPU-exact-test',physical_gpu_index=2)
        observed.append(dict(env))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gpu_lock.subprocess,'run',child)
    assert gpu_lock.main()==0
    assert observed[0]=={'CUDA_VISIBLE_DEVICES':'2','AUTODL_PHYSICAL_GPU_INDEX':'2','AUTODL_PHYSICAL_GPU_UUID':'GPU-exact-test'}


def test_llm_specific_owner_still_pins_uuid():
    from src.ablations.llm import existing_gpu_owner
    source=Path(existing_gpu_owner.__file__).read_text()
    assert 'CUDA_VISIBLE_DEVICES=sampler.uuid' in source
