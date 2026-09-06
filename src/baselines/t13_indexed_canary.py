"""Short real T13 eager-batch/lazy-batch update and resume parity, not full science."""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from src.baselines.t13_indexed_augmentation import _eager_without_split, _tensor_digest, restore_mask
from src.baselines.t13_component_diagnostics import cpu_copy, exact_difference, strict_diagnostic_runtime
from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256, sha256_file


class T13IndexedCanaryComplete(RuntimeError):
    def __init__(self, report):
        self.report=report
        super().__init__('T13_INDEXED_TRAINING_CANARY_COMPLETE')


class T13CanaryParityError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('T13_COMPONENT_DIAGNOSTIC_NOT_PASS:' + str(report.get('first_difference')))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all(value['cuda'])


def memory_snapshot(phase):
    sample=dict(phase=phase,monotonic_seconds=time.monotonic(),
        max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):sample[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
        for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt'):
            sample[name]=int((Path('/sys/fs/cgroup/memory')/name).read_text())
    except (OSError,ValueError):pass
    if torch.cuda.is_available():
        sample.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return sample


def state_digest(value):
    if torch.is_tensor(value):
        return _tensor_digest(value)
    if isinstance(value,np.ndarray):
        return hashlib.sha256(str((str(value.dtype),value.shape)).encode()+value.tobytes()).hexdigest()
    if isinstance(value,dict):
        return stable_sha256({str(k):state_digest(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)):
        return stable_sha256([state_digest(v) for v in value])
    return stable_sha256(value)


def eager_batch(dataset, indices):
    """Bounded official eager tensorization and original __getitem__ for one batch."""
    parents=[dataset._parent(int(dataset.graph_idx_list[i])) for i in indices]
    cls=_eager_without_split(importlib.import_module('data.dataset').AugmentedDataset)
    eager=cls(dataset.dataset, torch.stack([r['feature'].argmax(-1) for r in parents]),
        torch.stack([r['adj'] for r in parents]),
        torch.stack([r['edge_attr'].argmax(-1) for r in parents]) if dataset.edge_attr_dim else None,
        torch.stack([r['label'] for r in parents]),torch.stack([r['num_nodes'] for r in parents]),
        torch.stack([r['num_edges'] for r in parents]),dataset.graph_idx_list[indices].long(),
        dataset.fs_idx_list[indices].long(),torch.stack([restore_mask(dataset.mask_axes[i]) for i in indices]))
    rows=[]
    for local,index in enumerate(indices):
        row=eager[local];row['index']=index;rows.append(row)
    return default_collate(rows)


def checkpoint_reopen(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    required={'model','optimizer','scheduler','rng','sampler','index_identity','resume_identity'}
    if set(payload)!=required:
        raise ValueError('T13_CANARY_CHECKPOINT_COMPONENTS')
    return dict(state='PASS',checkpoint_sha256=sha256_file(path),state_sha256=state_digest(payload),
        components=sorted(required),model_training=False,ot_recomputed=False)


def run_training_parity(*,model,fss,train_loader,learning_rate,output_root,resume_identity,
                        diagnostic_profile='native'):
    with strict_diagnostic_runtime(diagnostic_profile) as numeric_contract:
        return _run_component_parity(model=model,fss=fss,train_loader=train_loader,
            learning_rate=learning_rate,output_root=output_root,resume_identity=resume_identity,
            diagnostic_profile=diagnostic_profile,numeric_contract=numeric_contract)


def _run_component_parity(*,model,fss,train_loader,learning_rate,output_root,resume_identity,
                          diagnostic_profile,numeric_contract):
    output=Path(output_root)
    output.mkdir(parents=True,exist_ok=False)
    dataset=train_loader.dataset.dataset
    if train_loader.batch_size!=500 or train_loader.num_workers!=0:
        raise ValueError('T13_CANARY_ORIGINAL_LOADER_CONTRACT')
    initial=cpu_copy(model.state_dict());initial_rng=rng_state()
    indices=list(train_loader.dataset.indices)
    from src.baselines.globalgce_resumable import _atomic_torch_save
    records=[];comparisons=[];started=time.monotonic();memory=[];initials=[]
    raw={'initial_model':initial,'initial_rng':cpu_copy(initial_rng),'fss':cpu_copy(fss),
         'dataset_identity':dataset.identity,'numeric_contract':numeric_contract,'records':records}
    first_difference=None
    execution_error=None
    current_phase='INITIALIZED'

    def report_payload():
        eager_controls=[row for row in comparisons if row['kind']=='EAGER_SELF_CONTROL']
        eager_done={row['arm'] for row in records if row['mode']=='eager' and row['epoch']==0 and 'after_update' in row}
        eager_repeatable=len(eager_done)==3 and len(eager_controls)==2 and all(row['exact'] for row in eager_controls)
        lazy=[row for row in comparisons if row['kind']=='EAGER_LAZY']
        good=bool(execution_error is None and eager_repeatable and len(lazy)==2
                  and all(row['exact'] for row in lazy) and (output/'checkpoint_reload.json').is_file())
        return dict(state='T13_COMPONENT_DIAGNOSTIC_PASS' if good else 'T13_COMPONENT_DIAGNOSTIC_INCOMPLETE_OR_FAILED',
            diagnostic_profile=diagnostic_profile,numeric_contract=numeric_contract,
            eager_repetitions_required=3,eager_repetitions_completed=len(eager_done),eager_self_repeatable=eager_repeatable,
            comparisons=comparisons,first_difference=first_difference,execution_error=execution_error,
            current_phase=current_phase,initial_state_bindings=initials,
            raw_evidence_path=str(output/'component_evidence.pt'),tolerance_used=False,
            warn_only_accepted=False,full_science_started=False,
            formal_numeric_contract_changed=False,
            deterministic_diagnostic_not_formal_approval=diagnostic_profile=='deterministic')

    def persist():
        # Two bounded files irrespective of parent/sample count. No expanded batch is saved.
        _atomic_torch_save(torch,raw,output/'component_evidence.pt')
        atomic_json(output/'component_diagnostics.json',report_payload())
    def sample(phase):
        memory.append(memory_snapshot(phase))
        atomic_json(output/'memory_boundaries.json',{'samples':memory})
    def new_optimizer():
        opt=torch.optim.Adam(model.parameters(),lr=float(learning_rate),weight_decay=1e-5)
        return opt,torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.9)
    def snapshot(opt,sched,next_epoch):
        return cpu_copy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            rng=rng_state(),sampler=dict(dataset.identity['sampler'],next_epoch=next_epoch),
            index_identity=dataset.identity,resume_identity=resume_identity))
    def step(opt,sched,mode,arm,epoch):
        nonlocal current_phase
        row=dict(mode=mode,arm=arm,epoch=epoch,batches=[],before=snapshot(opt,sched,epoch))
        records.append(row)
        current_phase=f'{arm}:epoch{epoch}:before_rules'
        sample(current_phase)
        model.train();model.gt_gnn.eval()
        rules=model.get_rules(fss)
        row['rules']=cpu_copy(rules)
        row['rng_after_rules']=cpu_copy(rng_state())
        # Both arms create exactly one identical iterator (same RNG draw).
        values=[0.0,0.0,0.0,0.0]
        for batch_index,lazy in enumerate(train_loader):
            # Exactly the pinned loop, including fetching then discarding batch6.
            if batch_index>=5:break
            if mode=='eager':
                batch=eager_batch(dataset,indices[batch_index*500:(batch_index+1)*500])
                if state_digest(batch)!=state_digest(lazy):
                    raise ValueError('T13_CANARY_BATCH_TENSOR_DIFFERENCE')
            else:
                batch=lazy
            current_phase=f'{arm}:epoch{epoch}:batch{batch_index}'
            batch_record=dict(batch_index=batch_index,batch_sha256=state_digest(batch),
                              rng_before=cpu_copy(rng_state()))
            sample(current_phase+':materialized')
            current=model.run_one_batch(rules,batch)
            batch_record.update(losses=cpu_copy(list(current)),rng_after=cpu_copy(rng_state()))
            row['batches'].append(batch_record)
            values=[left+right for left,right in zip(values,current,strict=True)]
            sample(current_phase+':forward')
        measured=[float(v.detach().cpu()) for v in values]
        if not all(np.isfinite(measured)):
            raise ValueError('T13_CANARY_NONFINITE_LOSS')
        row['losses']=cpu_copy(values)
        current_phase=f'{arm}:epoch{epoch}:backward'
        values[3].backward();sample(current_phase)
        row['gradients']={name:cpu_copy(parameter.grad) for name,parameter in model.named_parameters()}
        row['rng_after_backward']=cpu_copy(rng_state())
        opt.step();opt.zero_grad();sched.step();sample(arm+':after_update')
        row['after_update']=snapshot(opt,sched,epoch+1)
        row['loss_values']=measured
        persist()
        return row

    def compare(reference,observed,kind):
        nonlocal first_difference
        # Names of the arms are metadata, never part of the scientific comparison.
        fields=('before','rules','rng_after_rules','batches','losses','gradients','rng_after_backward','after_update')
        components={field:exact_difference(reference[field],observed[field]) for field in fields}
        result=dict(kind=kind,reference_arm=reference['arm'],observed_arm=observed['arm'],epoch=observed['epoch'],
                    exact=all(value['exact'] for value in components.values()),components=components)
        comparisons.append(result)
        if not result['exact'] and first_difference is None:
            field=next(field for field in fields if not components[field]['exact'])
            first_difference=dict(kind=kind,component=field,reference_arm=reference['arm'],
                observed_arm=observed['arm'],epoch=observed['epoch'],**components[field]['differences'][0])
        persist()
        return result['exact']

    try:
        reference=[]
        # Three independent eager starts are mandatory even if repeat #2 differs.
        for arm,mode,steps in (('eager_0','eager',2),('eager_1','eager',1),('eager_2','eager',1),('lazy','lazy',2)):
            model.load_state_dict(initial);model.zero_grad(set_to_none=True);restore_rng(initial_rng)
            opt,sched=new_optimizer()
            binding=dict(arm=arm,state_sha256=state_digest(snapshot(opt,sched,0)))
            initials.append(binding)
            if binding['state_sha256']!=initials[0]['state_sha256']:
                raise ValueError('T13_ARM_INITIAL_STATE_CHANGED')
            for epoch in range(steps):
                row=step(opt,sched,mode,arm,epoch)
                if arm=='eager_0':
                    reference.append(row)
                else:
                    equal=compare(reference[epoch],row,'EAGER_SELF_CONTROL' if mode=='eager' else 'EAGER_LAZY')
                    if mode=='lazy' and not equal:
                        break
                if mode=='lazy' and epoch==0:
                    checkpoint=output/'training_checkpoint.pt'
                    current=row['after_update']
                    sample('lazy:before_checkpoint_save');_atomic_torch_save(torch,current,checkpoint);sample('lazy:after_checkpoint_save')
                    receipt=checkpoint_reopen(checkpoint)
                    if receipt['state_sha256']!=state_digest(current):
                        raise ValueError('T13_CANARY_SERIALIZED_STATE_DRIFT')
                    loaded=torch.load(checkpoint,map_location=model.device,weights_only=False)
                    sample('lazy:after_checkpoint_load')
                    model.load_state_dict(loaded['model']);opt.load_state_dict(loaded['optimizer']);sched.load_state_dict(loaded['scheduler'])
                    restored=loaded['rng'];restored['torch']=restored['torch'].cpu()
                    if restored['cuda'] is not None:restored['cuda']=[value.cpu() for value in restored['cuda']]
                    restore_rng(restored)
                    sample('lazy:after_model_optimizer_rng_restore')
                    atomic_json(output/'checkpoint_reload.json',receipt)
    except Exception as error:
        execution_error=dict(type=type(error).__name__,message=str(error),phase=current_phase)
        persist()
        raise T13CanaryParityError(report_payload()) from error
    diagnostics=report_payload()
    persist()
    if diagnostics['state']!='T13_COMPONENT_DIAGNOSTIC_PASS':
        raise T13CanaryParityError(diagnostics)
    report=dict(state='T13_INDEXED_SHORT_TRAINING_PARITY_PASS',dataset_identity=dataset.identity,
        real_model_class=type(model).__name__,eager_lazy_batch_exact=True,forward_loss_exact=True,
        model_optimizer_scheduler_rng_exact=True,checkpoint_reload_exact=True,
        optimizer_updates_per_arm=2,eager_repetitions=3,diagnostic_profile=diagnostic_profile,
        numeric_contract=numeric_contract,eager_self_repeatable=True,
        component_diagnostics_sha256=sha256_file(output/'component_diagnostics.json'),
        component_evidence_sha256=sha256_file(output/'component_evidence.pt'),
        batches_per_update=min(5,len(train_loader)),batch_size=train_loader.batch_size,
        full_original_batches_per_epoch=5,short_diagnostic_only=True,full_science_started=False,
        sample_order_multiplicity_changed=False,training_config_changed=False,
        losses=[row['loss_values'] for row in reference],checkpoint_sha256=sha256_file(output/'training_checkpoint.pt'),
        memory_boundary_samples=memory,
        elapsed_seconds=time.monotonic()-started,max_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    atomic_json(output/'training_parity.json',report)
    return report
