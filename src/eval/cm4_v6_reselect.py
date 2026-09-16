"""Small CM4 adapter around the accepted V5 raw-matrix evaluator.

No model/generator actions. Original calibration matrices and complete pools are
reused, and missing selected test pairs alone use original saved embeddings.
"""
from pathlib import Path
from datetime import datetime,timezone
import os
import numpy as np
from src.eval.taste_final_v5_cm import TasteCM,write_csv
from src.baselines.cm_crem_runtime import read_json,digest,file_sha,atomic_json,require_compute_node
from src.baselines.cm_crem_selection import SelectionFreeze
from src.baselines.cm_crem_k20 import freeze_k20

DEADLINE='2026-09-16T16:27:44+00:00'


class CM4(TasteCM):
    def __init__(self,dataset,source,output):
        if dataset not in ('AIDS','Mutagenicity','BACE'):raise ValueError('Taste V5 is adoption-only')
        self.dataset=dataset;self.source=Path(source);self.root=Path(output);self.enc={};self.pred={};self.parent_meta={}
        self.old_spec=read_json(self.source/'spec.json');self.bace=dataset=='BACE'
        if self.bace:
            self.pilot=read_json(self.old_spec['source_spec']);self.legacy=Path(self.old_spec['source_root'])
            self.audit_old=read_json(self.source/'audit/k20_audit.json')
            if self.audit_old['status']!='K20_RECORDS_AND_INDEPENDENT_PROTOTYPE_CHECKS_PASS':raise ValueError('BACE acceptance')
            self.old_sha=self.audit_old['scope_sha256'];pool=read_json(self.source/'pool.json')
            self.old_f=SelectionFreeze.from_dict(read_json(self.source/'freeze.json')['freeze']);self.pool_ids=pool['ids']
            self.pool_pred={r['candidate_id']:r for r in pool['candidates']}
            grid=self.pilot['resolved_evaluation']['threshold_grid'];model=self.pilot['resolved_oracle']['model_sha256']
            temp=self.pilot['resolved_oracle']['temperature'];distance=self.pilot['resolved_wnode']
            self.oracle_binding=dict(weight=model,temperature=temp,source_spec_sha=file_sha(self.old_spec['source_spec']))
        else:
            self.old_sha=digest({k:v for k,v in self.old_spec.items() if k not in {'execution_commit','output_root'}})
            self.audit_old=self.old('audit/final_audit.json')
            if self.audit_old['status']!='CM_DATASET_POSTFILTER_AUDIT_PASS' or self.audit_old['fixture'] or not self.audit_old['scientific_pass_claimed']:raise ValueError('Independent original acceptance')
            if file_sha(self.old_spec['pilot_spec'])!=self.old_spec['pilot_spec_sha256']:raise ValueError('Original oracle spec changed')
            self.pilot=read_json(self.old_spec['pilot_spec']);model=self.pilot['oracle_sha256'];temp=None;distance=self.pilot['resolved_wnode']
            self.old_f=SelectionFreeze.from_dict(self.old('selection_freeze.json'));self.pool_ids=self.old('pool_binding.json')['candidate_ids']
            grid=self.old_spec['evaluation']['grid'];self.oracle_binding=dict(weight=model,pilot_sha=self.old_spec['pilot_spec_sha256'])
        if list(self.old_f.pool_candidate_ids)!=self.pool_ids:raise ValueError('Complete accepted pool/order required')
        scope='SOURCE-DESCRIPTIVE-v1_NON_HELDOUT' if dataset=='AIDS' else 'ORIGINAL_HELDOUT_FIXED_BASE'
        scope_payload=read_json(self.source/'scope_contract.json') if dataset=='AIDS' else None
        if scope_payload and (scope_payload['heldout'] or scope_payload['selection_count']!=220 or scope_payload['evaluation_count']!=1283):raise ValueError('AIDS scope changed')
        self.c=dict(experiment_id='CM4_TASTE_K20_THETA010_CLOSEOUT_V6',dataset=dataset,scope=scope,
            scope_evidence_sha=digest(scope_payload),primary_theta=.1,primary_report_k=20,k_max=20,cost_cap=self.old_f.cap,
            oracle_sha256=model,temperature=temp,source_label=1,allowed_destinations=[0],distance=distance,
            oracle_binding=self.oracle_binding,calibration_ids=list(self.old_f.calibration_parent_ids),
            source_calibration=sum(self.old_f.calibration_source_mask),source_spec_sha=file_sha(self.source/'spec.json'),
            source_acceptance_sha=digest(self.audit_old),grid=grid,deadline=DEADLINE,post_hoc=True,
            new_generation=False,main_authority_written=False)
        self.sha=digest(self.c)

    def admission(self):
        require_compute_node()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('HPC CPU-only')
        if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):raise RuntimeError('ORIGINAL_DEADLINE')
        if not self.root.is_absolute() or not self.root.is_relative_to('/share/home/u20526/czx'):raise ValueError('HPC scope')
        self.root.mkdir(parents=True,exist_ok=True)
        if os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize<2*1024**3:raise ValueError('Output capacity')

    def matrix_old(self,split):
        if not self.bace:
            meta=self.old(split+'_prepared.json');self.c[split+'_ids']=meta['parent_ids'];self.c['source_'+split]=sum(meta['source_mask'])
            return super().matrix_old(split)
        if split=='calibration':parents=list(self.old_f.calibration_parent_ids);cols=self.pool_ids
        else:
            meta=read_json(self.source/'test_evaluation.json');parents=meta['parent_ids'];cols=list(self.old_f.selected_candidate_ids)
        d=np.full((len(parents),len(cols)),np.nan);s=np.zeros(d.shape,dtype=np.uint8);mask=[]
        for i,pid in enumerate(parents):
            r=read_json(self.source/split/(digest(pid)[:20]+'.json'))
            if r['scope_sha256']!=self.old_sha or r['parent_id']!=pid or r['candidate_ids']!=cols:raise ValueError('BACE pair matrix binding')
            src=r['prediction']['predicted_label']==1;mask.append(src)
            if split=='test':self.parent_meta[pid]=r
            for j,x in enumerate(r['pairs']):
                if x['candidate_id']!=cols[j] or x['parent_id']!=pid or x['pair_status']!=('OK' if src else 'BEFORE_NOT_SOURCE'):raise ValueError('BACE pair identity/status')
                d[i,j]=float(x['distance']) if src else np.inf;s[i,j]=1 if src else 2
        self.c[split+'_ids']=parents;self.c['source_'+split]=sum(mask)
        return d,s,parents,cols,np.asarray(mask,dtype=bool)

    def select(self):
        self.admission();self.put('contract.json',self.c)
        if (self.root/'selection_freeze.json').exists():self.freeze();return
        d,s,p,ids,mask=self.matrix_old('calibration')
        if ids!=self.pool_ids or p!=list(self.old_f.calibration_parent_ids):raise ValueError('Full original calibration required')
        from src.baselines.cm_crem_postfilter import decode_status
        f,r=freeze_k20(d,decode_status(d,s),p,ids,mask,.1,self.c['cost_cap'],self.c['grid'],self.sha,self.old_f.frozen_pool_sha256)
        self.put('selection_freeze.json',f.to_dict());self.put('selection_report.json',dict(**r,producer_pid=os.getpid(),
            new_test_read_before_freeze=False,previous_test_observed=True,overlap_old=len(set(f.selected_candidate_ids)&set(self.old_f.selected_candidate_ids))))
        print(self.dataset,'FROZEN',r['objective'],flush=True)

    def load_encoding(self,role,wanted=None):
        if self.bace:
            if role=='test':
                for pid,r in self.parent_meta.items():
                    self.pred[pid]=r['prediction']
                    if r['parent_encoding']:self.enc[r['prediction']['full_graph_id']]=r['parent_encoding']
            else:
                self.pred.update(self.pool_pred)
                paths=[self.legacy/'pool_encodings.json',*sorted((self.source/'encodings').glob('block-*.json'))]
                for path in paths:
                    for r in read_json(path)['records']:
                        if wanted is None or r['candidate_id'] in wanted:self.enc[r['candidate_id']]={**r,'H':np.asarray(r['H'])}
            return
        for path in sorted((self.source/'encoding').glob(role+'-*.json')):
            b=self.old(str(path.relative_to(self.source)))
            for r in b['predictions']:
                if r['oracle_weight_sha256']!=self.c['oracle_sha256']:raise ValueError('RF oracle changed')
                self.pred[r.get('parent_id',r.get('candidate_id'))]=r
            rows=b['records']
            if wanted is not None and not any(r['candidate_id'] in wanted for r in rows):continue
            if file_sha(path.with_suffix('.npz'))!=b['npz_sha256']:raise ValueError('Encoding bytes changed')
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                for i,r in enumerate(rows):
                    if wanted is None or r['candidate_id'] in wanted:self.enc[r['candidate_id']]={**r,'H':z[f'H{i}'].copy()}

    def audit(self):
        self.admission();f=self.freeze();self.matrix_old('test')
        r=read_json(self.root/'selection_report.json')
        if r['producer_pid']==os.getpid():raise ValueError('Independent process required')
        d,s,p,c,m=self.matrix_old('calibration')
        from src.baselines.cm_crem_postfilter import decode_status
        replay,_=freeze_k20(d,decode_status(d,s),p,c,m,.1,self.c['cost_cap'],self.c['grid'],self.sha,self.old_f.frozen_pool_sha256)
        if replay!=f:raise ValueError('Independent selection replay failed')
        result=self.result();new=set(f.selected_candidate_ids)-set(self.old_f.selected_candidate_ids);checks=[]
        if new:
            self.load_encoding('pool',new);self.load_encoding('test')
            for b,start in enumerate(range(0,len(self.c['test_ids']),8)):
                path=self.root/f'test/block-{b:04d}';meta=read_json(path.with_suffix('.json'))
                if meta['producer_pid']==os.getpid():raise ValueError('Independent pair process required')
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                    for i,pid in enumerate(meta['parent_ids']):
                        for j,cid in enumerate(meta['candidate_ids']):
                            if cid not in new or z['states'][i,j]!=1 or len(checks)>=16:continue
                            left=self.enc[self.pred[pid]['full_graph_id']];right=self.enc[self.pred[cid]['full_graph_id']]
                            actual=self.exact(left,right)
                            if actual!=z['values'][i,j]:raise ValueError('Independent OT differs')
                            checks.append(dict(parent=pid,candidate=cid,raw=actual,parent_graph=left['graph_sha256'],prototype_graph=right['graph_sha256']))
                if len(checks)>=16:break
        self.put('test_evaluation.json',result.to_dict())
        self.put('audit/final_audit.json',dict(status='CM4_V6_ACCEPTED',dataset=self.dataset,scope=self.c['scope'],
            contract_sha256=self.sha,original_acceptance=self.audit_old,source_models_repeated=False,
            independent_new_pairs=checks,metrics_k20=result.prefix_metrics()[-1],main_authority_written=False))
        write_csv(self.root/'source_csv/prefix_metrics.csv',result.prefix_metrics())
        write_csv(self.root/'source_csv/parent_best_distances.csv',result.parent_best_rows())
        write_csv(self.root/'source_csv/exact_ecdf_k20.csv',result.exact_ecdf(20))


def first_prototype_audit(contract,source,v5cm,output):
    x=TasteCM(contract,source,v5cm);x.admission();f=x.freeze();cid=f.selected_candidate_ids[0]
    x.load_encoding('pool',{cid});x.load_encoding('test');raw=[]
    from rdkit import Chem
    after=x.pred[cid];right=x.enc[after['full_graph_id']]
    if after['predicted_label'] not in [0,2] or Chem.MolFromSmiles(right['canonical_smiles']) is None:raise ValueError('Illegal/non-target prototype')
    for b,start in enumerate(range(0,len(x.c['test_ids']),8)):
        path=x.root/f'test/block-{b:04d}';m=read_json(path.with_suffix('.json'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            for i,pid in enumerate(m['parent_ids']):
                if not m['source_mask'][i]:continue
                before=x.pred[pid];left=x.enc[before['full_graph_id']]
                if before['predicted_label']!=1 or left['graph_sha256']==right['graph_sha256']:raise ValueError('Source/self-pair conflict')
                raw.append(dict(parent_id=pid,candidate_id=cid,parent_graph_sha=left['graph_sha256'],prototype_graph_sha=right['graph_sha256'],
                    parent_encoding_sha=left['encoding_sha256'],prototype_encoding_sha=right['encoding_sha256'],raw=float(z['values'][i,0])))
    ordered=sorted(raw,key=lambda r:digest(['V6_FIRST_PROTOTYPE_AUDIT',r['parent_id']]))[:8];checked=[]
    for r in ordered:
        actual=x.exact(x.enc[x.pred[r['parent_id']]['full_graph_id']],right)
        if actual!=r['raw']:raise ValueError('Independent first-column raw differs')
        checked.append({**r,'independent_raw':actual,'exact_equal':True})
    if len(raw)!=285 or len(checked)!=8:raise ValueError('Incomplete source or audit cohort')
    atomic_json(Path(output)/'first_prototype_audit.json',dict(status='FIRST_PROTOTYPE_GRAPH_AND_8_EXACT_PAIRS_PASS',
        prototype_prediction=after,prototype_graph={k:v for k,v in right.items() if k!='H'},
        source_pairs=raw,checked_pairs=checked,first_max=max(r['raw'] for r in raw),new_ot_calls=8,
        adopted_v5_new_pairs_audit=16,source_count=285,base_count=468,source_ceiling=285/468))
