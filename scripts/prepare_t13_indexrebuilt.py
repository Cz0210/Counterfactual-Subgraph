#!/usr/bin/env python3
"""CPU-only bounded reconstruction of lost T13 compact layout, not formal training.

Loads the actual epoch29 model/optimizer and original adopted patterns. The
result still requires GPU real-batch/reload, original lease and durable backup
before same-run continuation. No test, mining, optimizer update or GPU usage.
"""
import argparse, hashlib, json, os, signal, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class Prepared(Exception): pass

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--plan',required=True)
    a=p.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('ISOLATED_MINUS_I_MINUS_B_REQUIRED')
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['output_root'])
    if out.exists():raise ValueError('FRESH_PREPARATION_ROOT_REQUIRED')
    if plan['scope']!='T13_INDEX_REBUILT_CPU_PREPARATION' or plan['formal_quota_used']!='1/1':raise ValueError('PLAN_SCOPE')
    if not out.is_relative_to(Path('/root/autodl-tmp')):raise ValueError('EXPLICIT_INDEPENDENT_NVME_REQUIRED')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_PREPARATION_MUST_HIDE_ALL_GPUS')
    from src.eval.bace_frozen_gnn_contracts import atomic_json
    from src.baselines.t13_real_batch_performance import snapshot_checkpoint
    start=time.monotonic();out.mkdir()
    def sample(stage):
        usage=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text());limit=int(Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())
        status=Path('/proc/self/status').read_text().splitlines();rss=next(int(x.split()[1])*1024 for x in status if x.startswith('VmRSS:'));peak=next(int(x.split()[1])*1024 for x in status if x.startswith('VmHWM:'))
        sv=os.statvfs(out);free=sv.f_bavail*sv.f_frsize
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-start,rss_bytes=rss,peak_bytes=peak,cgroup_usage=usage,cgroup_limit=limit,nvme_free_bytes=free,formal_started=False,gpu_used=False)
        atomic_json(out/'progress.json',row)
        if peak>plan['process_peak_bound_bytes']:raise ValueError('CPU_PREPARATION_PROCESS_PEAK')
        if limit-usage < plan['other_remaining_reserve_bytes']+max(0,plan['process_peak_bound_bytes']-rss):raise ValueError('CPU_PREPARATION_INCREMENTAL_HEADROOM')
        if free < plan['nvme_minimum_remaining_bytes']+plan['remaining_output_bound_bytes']:raise ValueError('CPU_PREPARATION_NVME_PEAK')
        return row
    def timeout(*_):raise TimeoutError('BOUNDED_CPU_PREPARATION_WALL_LIMIT')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['max_wall_seconds'])
    try:
        sample('before_checkpoint_load')
        import torch
        torch.set_num_threads(plan['torch_num_threads'])
        torch.set_num_interop_threads(plan['torch_num_interop_threads'])
        snap=snapshot_checkpoint(plan['source_checkpoint'],out/'source_checkpoint.pt')
        checkpoint=torch.load(snap['snapshot'],map_location='cpu',weights_only=False)
        old=json.loads(Path(plan['source_index_manifest']).read_text())
        ledger=json.loads(Path(plan['formal_ledger']).read_text())
        from src.baselines.globalgce_resumable import validate_globalgce_epoch_checkpoint_identity, _get_fs_expanded_data_from_adoption, _atomic_torch_save
        validate_globalgce_epoch_checkpoint_identity(checkpoint,checkpoint['resume_identity'])
        if old!=checkpoint['augmented_dataset_identity']:raise ValueError('SOURCE_INDEX_NOT_BOUND_TO_CHECKPOINT')
        atomic_json(out/'source_checkpoint_receipt.json',dict(snap,epoch=checkpoint['epoch'],next_epoch=checkpoint['next_epoch'],model_tensor_count=len(checkpoint['model_state']),optimizer_state_count=len(checkpoint['optimizer_state']['state']),formal_quota_used='1/1',old_checkpoint_unchanged=True))
        sample('source_checkpoint_loaded')
        from src.baselines.tastemolnet_globalgce_full import _checkpoint_payloads,load_full_train_split,select_full_sweet_train_cohort,FrozenTasteGINEScorer
        from src.baselines.globalgce_bace_native_rules import validate_official_globalgce_root
        from src.baselines import globalgce_mutagenicity_adapter as adapter
        official=validate_official_globalgce_root(Path(plan['official_root']))
        payloads=_checkpoint_payloads(Path(plan['gnn_checkpoint']));split=json.loads(payloads['split_manifest.json'])
        if hashlib.sha256(Path(plan['train_csv']).read_bytes()).hexdigest()!=split['files']['train']['sha256']:raise ValueError('TRAIN_INPUT_CHANGED')
        train=load_full_train_split(SimpleNamespace(train_path=Path(plan['train_csv']),train_count=split['train_manifest']['num_records'],train_label_counts=split['train_manifest']['label_counts']))
        scorer=FrozenTasteGINEScorer(payloads,device='cpu',batch_size=256)
        selected,cohort=select_full_sweet_train_cohort(train,scorer=scorer,batch_size=256)
        if cohort!=json.loads(Path(plan['source_cohort_manifest']).read_text()):raise ValueError('ORIGINAL_SOURCE_COHORT_CHANGED')
        atomic_json(out/'train_cohort_manifest.json',cohort);del scorer
        sample('same_source_cohort_reloaded')
        generator=adapter.OfficialGlobalGCEMutagenicityGenerator(Path(plan['official_root']),native_train_csv=Path(plan['train_csv']),dataset_name='TasteMolNet',min_freq=2,frozen_gine_checkpoint=Path(plan['gnn_checkpoint']),source_label=1,target_label=0,num_classes=3,official_source_authority=official['runtime_source_authority'],require_isolated_imports=True,rules_only_min_valid_native_rules=0)
        generator.t13_indexed_options=dict(storage='t13_indexed_augmentation_v1',diagnostic_profile='deterministic')
        def intercept(**kwargs):
            from src.baselines.t13_index_rebuilt import rebuild_authorized,adapted_checkpoint
            from src.baselines.t13_bounded_payload import seal_payload,load_index
            model=kwargs['model']
            validate_globalgce_epoch_checkpoint_identity(checkpoint,kwargs['resume_identity'])
            def expansion(dataset,fs_dict,crop_expansion=False):
                if crop_expansion:raise ValueError('CROP_CONTRACT_CHANGED')
                return rebuild_authorized(base_dataset=dataset,fsg=model.fsg,fs_dict=fs_dict,expected_identity=old,private_seed=plan['private_index_rng_seed'])
            model.fsg.expand_data_by_fs=expansion
            sample('before_bounded_index_rebuild')
            (fss,tr,va,te),adoption=_get_fs_expanded_data_from_adoption(model=model,train_loader=kwargs['train_loader'],proof_path=kwargs['gspan_adoption_proof'])
            indexed=tr.dataset.dataset;sample('index_rebuilt')
            descriptor=seal_payload(indexed,out/'compact_index')
            restored=load_index(descriptor,base_dataset=indexed.dataset,fsg=model.fsg,expected_identity=indexed.identity)
            if restored.identity!=indexed.identity:raise ValueError('COMPACT_RELOAD_IDENTITY')
            model.load_state_dict(checkpoint['model_state'],strict=True)
            opt=torch.optim.Adam(model.parameters(),lr=kwargs['learning_rate'],weight_decay=1e-5);opt.load_state_dict(checkpoint['optimizer_state'])
            scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.9);scheduler.load_state_dict(checkpoint['scheduler_state'])
            for name,tensor in model.state_dict().items():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():raise ValueError('NONFINITE_SOURCE_MODEL:'+name)
            adapted,receipt=adapted_checkpoint(checkpoint,indexed.identity,original_identity=old,formal_ledger=ledger)
            _atomic_torch_save(torch,adapted,out/'index_rebuilt_working_checkpoint.pt')
            atomic_json(out/'index_rebuilt_receipt.json',dict(receipt,compact_payload=descriptor,adoption=adoption,cpu_model_optimizer_scheduler_loaded=True,gpu_batch_reload_pass=False))
            sample('working_checkpoint_saved')
            raise Prepared()
        previous=adapter.train_globalgce_resumable;adapter.train_globalgce_resumable=intercept
        try:
            generator.generate(selected,output_dir=out/'native_preparation',seed=7,epochs=100,top_k_native=20,learning_rate=.1,dropout=.5,device='cpu',resume=False,gspan_adoption_proof=Path(plan['gspan_adoption_proof']),rules_only=True)
        except Prepared:
            atomic_json(out/'terminal.json',dict(state='INDEX_REBUILT_CPU_PREPARED_NOT_FORMAL',formal_started=False,optimizer_updates=0,requires=['GPU_TWO_TRAIN_ONE_VALIDATION_AND_RELOAD','DURABLE_COMPACT_BACKUP','EXISTING_OWNER_LEASE_PROVIDER_CONTINUATION'],last_resource=sample('complete')))
            return 0
        finally:adapter.train_globalgce_resumable=previous
        raise ValueError('PREPARATION_DID_NOT_INTERCEPT_TRAINING')
    except Exception as exc:
        atomic_json(out/'terminal.json',dict(state='PREPARATION_FAILED',error_type=type(exc).__name__,error=str(exc),formal_started=False,optimizer_updates=0,elapsed_seconds=time.monotonic()-start))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':raise SystemExit(main())
