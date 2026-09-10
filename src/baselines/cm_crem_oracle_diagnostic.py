"""Bounded CM oracle evidence capture; no generation, OT, selection, or PASS."""
import copy
import os
import socket
import subprocess
from pathlib import Path
import numpy as np
from src.baselines.cm_crem_runtime import read_json, atomic_json, digest, require_compute_node, checked_root, utc_now
from src.baselines.cm_crem_oracle import FrozenCMOracle
from src.baselines.cm_crem_audit import fixed_spotcheck_pairs

def difference(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'exact':False,'invalid_shape_or_nonfinite':True}
    d=np.abs(a-b)
    return {'exact':bool(np.array_equal(a,b)),'max_abs':float(d.max()),
            'max_rel':float((d/np.maximum(np.abs(b),1e-300)).max()),
            'first_index':next((list(i) for i in np.ndindex(a.shape) if a[i]!=b[i]),None)}

def input_only(row):
    return {k:row[k] for k in ('parent_id','candidate_id','smiles') if k in row}

def diagnostic(spec_path,source_root,output_root):
    require_compute_node()
    import torch
    import rdkit
    spec=read_json(spec_path); root=Path(source_root)
    out=checked_root(output_root,'/share/home/u20526/czx');out.mkdir(parents=True,exist_ok=False)
    report={'status':'DIAGNOSTIC_RUNNING','started_at':utc_now(),'job_id':os.environ['SLURM_JOB_ID'],
      'host':socket.gethostname(),'pid':os.getpid(),'science_hash':spec['science_hash'],
      'producer_commit':spec['execution']['execution_commit'],
      'diagnostic_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'generation_calls':0,'ot_calls':0,'scientific_pass_claimed':False,
      'historical_tensor_snapshot':'NOT_SAVED_BY_PRODUCER','historical_per_call_backend':'NOT_SAVED',
      'reconstruction_is_historical_capture':False}
    atomic_json(out/'diagnostic.json',report)
    try:
        prov=read_json(root/'audit/provenance_review.json');freeze=read_json(root/'selection_freeze.json');pool=read_json(root/'pool_freeze.json')
        if prov['science_hash']!=spec['science_hash'] or prov['status']!='BACE_SAVED_RECORD_PROVENANCE_VERIFIED' or freeze['frozen_pool_sha256']!=pool['pool_sha256']:
            raise ValueError('Saved provenance/pool/freeze mismatch')
        if spec['resolved_oracle'].get('forward_atol',0) or spec['resolved_oracle'].get('forward_rtol',0):raise ValueError('Expected unchanged zero-tolerance original contract')
        rows=[read_json(root/n) for n in sorted(prov['bound_record_sha256']) if n.startswith('test/parents/') and n.endswith('.json')]
        sample=fixed_spotcheck_pairs(rows,freeze['selected_candidate_ids'],spec['science_hash'])
        prototypes={r['candidate_id']:r for r in pool['candidates']}
        oracle=FrozenCMOracle.from_resolved(spec)
        report['environment']={'torch':torch.__version__,'rdkit':rdkit.__version__,'numpy':np.__version__,
          'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
          'deterministic':torch.are_deterministic_algorithms_enabled(),'mkldnn':torch.backends.mkldnn.enabled,
          'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
          'device':str(oracle.oracle.device),'temperature':oracle.oracle.temperature,
          'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')}}
        attempts=[];focal=None
        for parent,pair in sample:
            for role,saved in (('parent',parent['prediction']),('prototype',prototypes[pair['candidate_id']])):
                fresh=oracle.predict_rows([input_only(saved)],split='independent_audit')[0]
                diff=difference(fresh['logits'],saved['logits'])
                attempts.append({'role':role,'parent_id':parent['parent_id'],'candidate_id':pair['candidate_id'],'saved':saved,'reconstructed_singleton':fresh,'difference':diff})
                if not diff['exact']:focal=(role,saved,fresh);break
            if focal:break
        report['audit_order_attempts']=attempts
        atomic_json(out/'diagnostic.json',report)
        if not focal:report['status']='BOUNDED_FAILURE_NOT_REPRODUCED';return report
        role,saved,fresh=focal
        report['first_failure']={'role':role,'identity':input_only(saved),'saved_logits':saved['logits'],'audit_logits':fresh['logits'],
          'saved_probabilities':saved['probabilities'],'audit_probabilities':fresh['probabilities'],
          'saved_z_over_T':(np.asarray(saved['logits'])/oracle.oracle.temperature).tolist(),
          'audit_z_over_T':(np.asarray(fresh['logits'])/oracle.oracle.temperature).tolist(),
          'label_changed':saved['predicted_label']!=fresh['predicted_label']}
        if role!='prototype':report['status']='PARENT_CONTEXT_REQUIRES_REVIEW';return report
        origin=saved['origins'][0]
        parent=next(p for p in read_json(root/'attribution.json')['parents'] if p['parent_id']==origin['parent_id'])
        generation=read_json(root/f"generation_units/{digest(parent['parent_id'])[:20]}.json")
        class Captured(Exception):pass
        inputs=[];original=oracle.predict_rows
        def capture(rows,*,split):
            if split=='train_generated':inputs.extend(copy.deepcopy(rows));raise Captured()
            return original(rows,split=split)
        oracle.predict_rows=capture
        try:oracle.filter_generated(parent,generation)
        except Captured:pass
        finally:oracle.predict_rows=original
        idx=next(i for i,r in enumerate(inputs) if r['candidate_id']==saved['candidate_id'])
        size=oracle.oracle.batch_size;start=idx//size*size;context=inputs[start:start+size];pos=idx-start
        graphs=[oracle._graph(r['smiles'],r['candidate_id'],'train_generated') for r in context]
        batch=next(iter(oracle.oracle._batches(graphs,size))).to(oracle.oracle.device)
        state={k:v.detach().cpu().clone() for k,v in oracle.oracle.model.state_dict().items()}
        torch.save({'batch':batch.cpu(),'model_and_buffers':state,'rng':torch.get_rng_state(),'context':context,'position':pos,
          'historical_saved_tensors':False},out/'controlled_snapshot.pt')
        def forward(model,b):
            with torch.no_grad():
                v=model(b)
                if isinstance(v,dict):v=v['logits']
                elif isinstance(v,tuple):v=v[-1]
                return v.detach().cpu().numpy().astype(float)[pos]
        repeats=[forward(oracle.oracle.model,batch) for _ in range(3)]
        second=FrozenCMOracle.from_resolved(spec)
        restored=torch.load(out/'controlled_snapshot.pt',map_location='cpu',weights_only=False)
        if not all(torch.equal(state[k],second.oracle.model.state_dict()[k].cpu()) for k in state):raise ValueError('Independent weight/buffer mismatch')
        reloads=[forward(second.oracle.model,restored['batch']) for _ in range(3)]
        refeaturized=second.predict_rows(context,split='train_generated')[pos]
        b2=next(iter(second.oracle._batches([second._graph(r['smiles'],r['candidate_id'],'train_generated') for r in context],size)))
        report['controlled_same_tensor']={'context_ids':[r['candidate_id'] for r in context],'position':pos,'origin':origin,
          'producer_repeats':[r.tolist() for r in repeats],'independent_reload_repeats':[r.tolist() for r in reloads],
          'repeats_exact':all(np.array_equal(repeats[0],v) for v in repeats+reloads),
          'saved_vs_reconstructed_batch':difference(repeats[0],saved['logits']),
          'refeaturized_prediction':refeaturized,
          'refeaturized_tensors_exact':{k:torch.equal(v.cpu(),b2[k].cpu()) for k,v in batch.to_dict().items() if torch.is_tensor(v)},
          'independent_weights_buffers_exact':True,
          'after_weights_buffers_exact':all(torch.equal(state[k],oracle.oracle.model.state_dict()[k].cpu()) for k in state)}
        report['status']='DIAGNOSTIC_CAPTURE_COMPLETE_NOT_ACCEPTANCE'
        report['next_stage']='REVIEW_CONTEXT_EVIDENCE_BEFORE_CORRECTION'
        return report
    except BaseException as exc:
        report.update(status='DIAGNOSTIC_ENGINEERING_FAILED',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['finished_at']=utc_now();atomic_json(out/'diagnostic.json',report)
