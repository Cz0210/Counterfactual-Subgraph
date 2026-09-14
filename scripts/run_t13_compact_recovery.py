#!/usr/bin/env python3
"""Real GPU capacity/reload probe of the completed T13 compact recovery pack.

Two train batches and one validation batch are diagnostics, not formal updates
or proof of the full five-batch/100-epoch trajectory. Original formal stays1/1.
"""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

class Complete(Exception):pass

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--spec');p.add_argument('--plan')
    p.add_argument('--action',choices=['owner','probe','continue','reload'],required=True);p.add_argument('--checkpoint');p.add_argument('--expected')
    a=p.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('ISOLATED_EXECUTION_REQUIRED')
    if a.action=='owner':
        from src.utils.t13_gap_recovery_owner import run
        return run(a.spec)
    if a.action=='reload':
        import torch
        from src.baselines.t13_indexed_canary import state_digest
        x=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state_digest(x)!=a.expected:raise ValueError('INDEPENDENT_RELOAD_DIFFERS')
        print(json.dumps(dict(state='INDEPENDENT_CONTAINER_RELOAD_PASS',pid=os.getpid())));return 0
    from src.ablations.llm.existing_gpu_owner import receive_owner_binding
    from src.eval.bace_frozen_gnn_contracts import atomic_json,sha256_file
    binding=receive_owner_binding();plan=json.loads(Path(a.plan).read_text());out=Path(plan['output_root']);out.mkdir(exist_ok=False)
    start=time.monotonic();samples=[]
    def sample(phase):
        from datetime import datetime,timezone
        from src.utils.final16_owner_registry_v1 import process_start_ticks
        e=json.loads(Path(binding['resource_live_evidence']).read_text())
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(e['observed_at'])).total_seconds()
        if not 0<=age<=120 or e.get('pause_requested') or not e.get('t13_admission',{}).get('allowed'):raise ValueError('FRESH_OWNER_RESOURCE_NOT_ADMITTED')
        if e['plan_sha256']!=sha256_file(a.plan) or os.environ['CUDA_VISIBLE_DEVICES']!=plan['gpu_uuid']:raise ValueError('PLAN_OR_UUID_CHANGED')
        for key in ['gpu_owner','gpu_child']:
            if process_start_ticks('/proc',e[key+'_pid'])!=e[key+'_start_ticks']:raise ValueError('OWNER_CHILD_IDENTITY_CHANGED')
        fd=binding['held_gpu_lock_fd'];s=os.fstat(fd);t=os.stat(binding['gpu_lock_path'])
        if (s.st_dev,s.st_ino)!=(t.st_dev,t.st_ino):raise ValueError('LEASE_IDENTITY_CHANGED')
        lock=json.loads(os.pread(fd,65536,0))
        if lock['gpu_child_pid']!=os.getpid() or lock['gpu_uuid']!=plan['gpu_uuid']:raise ValueError('LEASE_CHILD_BINDING')
        from src.baselines.t13_indexed_canary import memory_snapshot
        row=memory_snapshot(phase);row.update(elapsed_seconds=time.monotonic()-start)
        if row.get('VmHWM_bytes',0)>16*1024**3:raise ValueError('BOUNDED_HOST_PROBE_16GIB_EXCEEDED')
        samples.append(row);atomic_json(out/'memory_boundaries.json',dict(samples=samples))
    def timeout(*_):raise TimeoutError('T13_PROBE_3600_SECOND_BOUND')
    signal.signal(signal.SIGALRM,timeout)
    if a.action=='probe':signal.alarm(3600)
    updates=0
    try:
        sample('before_cuda')
        competitor=subprocess.run([sys.executable,'-I','-B','-c',
            'import fcntl,sys; f=open(sys.argv[1],"r+");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nelse: sys.exit(2)',binding['gpu_lock_path']])
        if competitor.returncode:raise ValueError('INDEPENDENT_LOCK_COMPETITOR_SUCCEEDED')
        import torch
        from src.utils.t13_performance_dispatch import source_backend
        helper=source_backend(plan);b=json.loads(Path(plan['source_runtime_backend_receipt']).read_text())
        if b['observed_backend']!=helper.BACKEND:raise ValueError('BACKEND_SOURCE_CHANGED')
        actual=helper.apply_backend(torch,helper.BACKEND)
        torch.set_num_threads(b['torch_num_threads']);torch.set_num_interop_threads(b['torch_num_interop_threads'])
        atomic_json(out/'runtime_backend_receipt.json',dict(actual_backend=actual,pid=os.getpid(),threads=torch.get_num_threads(),gpu_uuid=plan['gpu_uuid']))
        from src.baselines.globalgce_resumable import _get_fs_expanded_data_from_adoption,_atomic_torch_save,_restore_numpy_rng_state
        from src.baselines.t13_indexed_canary import restore_rng,rng_state,state_digest
        from src.baselines.t13_component_diagnostics import cpu_copy
        from src.baselines.t13_bounded_payload import install_committed_expansion
        checkpoint=torch.load(plan['working_checkpoint'],map_location='cpu',weights_only=False)
        if checkpoint['next_epoch']!=30:raise ValueError('NOT_EPOCH29_RECOVERY')
        from src.baselines.tastemolnet_globalgce_full import _checkpoint_payloads,load_full_train_split,select_full_sweet_train_cohort,FrozenTasteGINEScorer
        from src.baselines.globalgce_bace_native_rules import validate_official_globalgce_root
        from src.baselines import globalgce_mutagenicity_adapter as adapter
        official=validate_official_globalgce_root(Path(plan['official_root']))
        payloads=_checkpoint_payloads(Path(plan['gnn_checkpoint']));split=json.loads(payloads['split_manifest.json'])
        if sha256_file(plan['train_csv'])!=split['files']['train']['sha256']:raise ValueError('TRAIN_INPUT_CHANGED')
        train=load_full_train_split(SimpleNamespace(train_path=Path(plan['train_csv']),train_count=split['train_manifest']['num_records'],train_label_counts=split['train_manifest']['label_counts']))
        scorer=FrozenTasteGINEScorer(payloads,device='cuda:0',batch_size=256)
        selected,cohort=select_full_sweet_train_cohort(train,scorer=scorer,batch_size=256)
        if cohort!=json.loads(Path(plan['source_cohort_manifest']).read_text()):raise ValueError('COHORT_CHANGED')
        del scorer;sample('source_cohort_loaded_no_index_rebuild')
        generator=adapter.OfficialGlobalGCEMutagenicityGenerator(Path(plan['official_root']),native_train_csv=Path(plan['train_csv']),dataset_name='TasteMolNet',min_freq=2,frozen_gine_checkpoint=Path(plan['gnn_checkpoint']),source_label=1,target_label=0,num_classes=3,official_source_authority=official['runtime_source_authority'],require_isolated_imports=True,rules_only_min_valid_native_rules=0)
        generator.t13_indexed_options=dict(storage='t13_indexed_augmentation_v1',diagnostic_profile='deterministic')
        if a.action=='continue':
            from src.baselines.t13_compact_continuation import continue_branches
            return continue_branches(plan=plan,out=out,checkpoint=checkpoint,selected=selected,
                cohort=cohort,official=official,adapter=adapter,sample=sample,torch=torch)
        def intercept(**kw):
            nonlocal updates
            model=kw['model'];install_committed_expansion(model.fsg,descriptor=plan['committed_compact_payload'],expected_identity=checkpoint['augmented_dataset_identity'])
            (fss,tr,va,_),adoption=_get_fs_expanded_data_from_adoption(model=model,train_loader=kw['train_loader'],proof_path=kw['gspan_adoption_proof'])
            if tr.dataset.dataset.identity!=checkpoint['augmented_dataset_identity'] or tr.batch_size!=500 or tr.num_workers:raise ValueError('COMPACT_BATCH_IDENTITY')
            model.load_state_dict(checkpoint['model_state']);model.gt_gnn.eval()
            opt=torch.optim.Adam(model.parameters(),lr=kw['learning_rate'],weight_decay=1e-5);opt.load_state_dict(checkpoint['optimizer_state'])
            sch=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.9);sch.load_state_dict(checkpoint['scheduler_state'])
            _restore_numpy_rng_state(kw['numpy_module'],checkpoint['numpy_rng_state'])
            restore_rng(dict(python=checkpoint['python_rng_state'],numpy=kw['numpy_module'].random.get_state(),torch=checkpoint['torch_rng_state'].cpu(),cuda=[v.cpu() for v in checkpoint['cuda_rng_state']]))
            sample('compact_arrays_loaded')
            iterator=iter(tr);rows=[]
            for i in range(2):
                model.train();model.gt_gnn.eval();opt.zero_grad(set_to_none=True)
                rules=model.get_rules(fss);batch=next(iterator);sample('batch_assembled_'+str(i))
                values=model.run_one_batch(rules,copy.deepcopy(batch))
                if not all(torch.isfinite(v).all() for v in values):raise ValueError('NONFINITE_LOSS')
                values[0].backward();torch.cuda.synchronize();sample('backward_'+str(i))
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('NONFINITE_GRADIENT')
                opt.step();updates+=1;rows.append(dict(batch=state_digest(batch),loss=[float(v.detach().cpu()) for v in values]))
                atomic_json(out/'diagnostic_ledger.json',dict(train_batches=updates,formal_updates=0,formal_quota='1/1'))
                if i==0:
                    saved=cpu_copy(dict(model_state=model.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sch.state_dict(),rng=rng_state(),compact_identity=checkpoint['augmented_dataset_identity']))
                    path=out/'diagnostic_checkpoint.pt';_atomic_torch_save(torch,saved,path);sample('save')
                    subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--config',a.config,'--action','reload','--checkpoint',str(path),'--expected',state_digest(saved)],check=True)
                    reloaded=torch.load(path,map_location='cpu',weights_only=False)
                    model.load_state_dict(reloaded['model_state']);opt.load_state_dict(reloaded['optimizer_state']);sch.load_state_dict(reloaded['scheduler_state']);restore_rng(reloaded['rng']);sample('independent_reload')
            model.eval();model.gt_gnn.eval()
            with torch.no_grad():
                vb=next(iter(va));vals=model.run_one_batch(model.get_rules(fss),copy.deepcopy(vb))
            if not all(torch.isfinite(v).all() for v in vals):raise ValueError('NONFINITE_VALIDATION')
            torch.cuda.synchronize();sample('validation_batch_complete')
            atomic_json(out/'probe.json',dict(state='REAL_GPU_TWO_TRAIN_ONE_VALIDATION_RELOAD_PASS',train=rows,validation_losses=[float(v.cpu()) for v in vals],actual_gpu_peak_bytes=torch.cuda.max_memory_allocated(),formal_updates=0,formal_quota='1/1',full_validation_completed=False,formal_continuation_requires='ORIGINAL_FULL_DUE_VALIDATION_AND_FIVE_BATCH_TRAINING_OWNER',old_checkpoint_changed=False))
            raise Complete()
        old=adapter.train_globalgce_resumable;adapter.train_globalgce_resumable=intercept
        try:
            generator.generate(selected,output_dir=out/'native_preparation',seed=7,epochs=100,top_k_native=20,learning_rate=.1,dropout=.5,device='cuda:0',resume=False,gspan_adoption_proof=Path(plan['gspan_adoption_proof']),rules_only=True)
        except Complete:return 0
        finally:adapter.train_globalgce_resumable=old
        raise ValueError('DID_NOT_INTERCEPT_TRAINING')
    except Exception as e:
        atomic_json(out/'probe.json',dict(state='GPU_PROBE_FAILED',error=repr(e),diagnostic_updates=updates,formal_updates=0,phase=samples[-1] if samples else None));raise
    finally:signal.alarm(0)

if __name__=='__main__':raise SystemExit(main())
