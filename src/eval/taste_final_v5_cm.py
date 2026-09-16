"""Taste V5: adopt accepted CM raw blocks, reselect once, evaluate missing pairs.

No generator/model loading and no authority writes. The accepted batch32 oracle
and MolCLR artifacts remain read-only. Each new block is resumable and compact.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.baselines.cm_crem_runtime import atomic_json, digest, file_sha, read_json, require_compute_node, utc_now
from src.baselines.cm_crem_selection import SelectionFreeze, evaluate_frozen_test
from src.baselines.cm_crem_k20 import freeze_k20
from src.baselines.cm_crem_postfilter import commit_npz, decode_status


def validate_contract(c):
    if (c['experiment_id'] != 'TASTE_GINE_K20_THETA010_FINAL_V5'
            or c['primary_theta'] != 0.1 or c['final_selection_theta'] != 0.1
            or c['primary_export_theta'] != 0.1 or c['k_max'] != 20
            or c['primary_report_k'] != 20 or c['cost_cap'] != 0.03416003659645076
            or c['num_classes'] != 3 or c['source_label'] != 1
            or c['allowed_destinations'] != [0, 2]):
        raise ValueError('V5 explicit scientific contract differs')
    for split in ('calibration', 'test'):
        ids = c[split + '_ids']
        if len(ids) != 468 or len(set(ids)) != 468 or digest(ids) != c[split + '_ids_sha256']:
            raise ValueError('Full ordered cohort identity differs: ' + split)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class TasteCM:
    def __init__(self, contract, source, output):
        self.c = read_json(contract); validate_contract(self.c); self.sha = digest(self.c)
        self.source, self.root = Path(source), Path(output)
        if not self.root.absolute().is_relative_to('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/experiments/taste_k20_theta010_final_v5'):
            raise ValueError('New science output must remain in the scoped HPC experiment root')
        self.old_spec = read_json(self.source/'spec.json')
        self.old_sha = digest({k:v for k,v in self.old_spec.items() if k not in {'execution_commit','output_root'}})
        self.audit_old = self.old('audit/final_audit.json')
        if (self.audit_old['status'] != 'CM_DATASET_POSTFILTER_AUDIT_PASS'
                or self.audit_old['fixture'] or not self.audit_old['scientific_pass_claimed']):
            raise ValueError('Actual independent original acceptance required')
        p = read_json(self.old_spec['pilot_spec'])
        if (p['oracle_sha256'] != self.c['oracle_sha256'] or p['source_label'] != self.c['source_label']
                or p['allowed_destinations'] != self.c['allowed_destinations']
                or p['oracle_files']['temperature_scaling.json'] != self.c['temperature_receipt_sha256']
                or p['resolved_wnode']['molclr_checkpoint_sha256'] != self.c['distance']['molclr_checkpoint_sha256']
                or self.old_spec['evaluation']['cap'] != self.c['cost_cap']):
            raise ValueError('CM source oracle/temperature/distance/cap incompatible')
        for split in ('calibration','test'):
            if self.old_spec['parents'][split]['sha256'] != self.c[split+'_file_sha256']:
                raise ValueError('Source split content differs')
        self.enc = {}; self.pred = {}

    def old(self, name):
        value = read_json(self.source/name)
        if value.get('contract_sha256') != self.old_sha:
            raise ValueError('Old accepted contract binding differs: '+name)
        return value

    def put(self, name, value):
        atomic_json(self.root/name, value, immutable=True)

    def admission(self):
        require_compute_node()
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('HPC CPU-only required')
        if datetime.now(timezone.utc) >= datetime.fromisoformat(self.c['deadline']):
            raise RuntimeError('Original deadline reached at committed boundary')
        self.root.mkdir(parents=True, exist_ok=True)
        v = os.statvfs(self.root)
        if v.f_bavail*v.f_frsize < 2*1024**3: raise RuntimeError('HPC output margin insufficient')

    def matrix_old(self, split):
        meta = self.old(split+'_prepared.json'); n=len(meta['parent_ids']); m=len(meta['candidate_ids'])
        ids = self.c[split+'_ids']
        if set(meta['parent_ids']) != set(ids): raise ValueError('Different full base cohort')
        values=np.full((n,m),np.nan); states=np.zeros((n,m),dtype=np.uint8)
        for block,start in enumerate(range(0,n,self.old_spec['parent_block_size'])):
            name=f'{split}/block-{block:04d}'; r=self.old(name+'.json')
            stop=min(start+self.old_spec['parent_block_size'],n)
            if r['parent_ids'] != meta['parent_ids'][start:stop] or r['candidate_ids'] != meta['candidate_ids']:
                raise ValueError('Raw block row/column binding differs')
            if file_sha(self.source/(name+'.npz')) != r['npz_sha256']: raise ValueError('Changed raw block')
            with np.load(self.source/(name+'.npz'),allow_pickle=False) as z:
                values[start:stop]=z['values']; states[start:stop]=z['states']
        decode_status(values,states)
        ix=[meta['parent_ids'].index(x) for x in ids]
        mask=np.asarray(meta['source_mask'],dtype=bool)[ix]
        if int(mask.sum()) != self.c['source_'+split]: raise ValueError('Source identity count differs')
        return values[ix], states[ix], ids, meta['candidate_ids'], mask

    def select(self):
        self.admission()
        if (self.root/'selection_freeze.json').exists(): self.freeze(); return
        d,s,p,c,m=self.matrix_old('calibration'); pool=self.old('pool_binding.json')
        if c != pool['candidate_ids']: raise ValueError('Incomplete original calibration pool')
        freeze,report=freeze_k20(d,decode_status(d,s),p,c,m,0.1,self.c['cost_cap'],
                               self.old_spec['evaluation']['grid'],self.sha,pool['pool_sha256'])
        old=SelectionFreeze.from_dict(self.old('selection_freeze.json'))
        self.put('source_adoption.json',dict(source_root=str(self.source),old_contract_sha256=self.old_sha,
            old_acceptance=self.audit_old,old_acceptance_sha256=digest(self.audit_old),
            old_freeze_sha256=old.freeze_sha256,pool_sha256=pool['pool_sha256'],
            complete_calibration_pairs=int(d.size),new_oracle_calls=0,new_calibration_ot=0,
            original_selector='cm_crem_k20.freeze_k20',selector_grid=self.old_spec['evaluation']['grid'],
            new_contract_sha256=self.sha,source_generation_repeated=False))
        self.put('selection_report.json',{**report,'same_order_as_old':freeze.selected_candidate_ids==old.selected_candidate_ids,
            'selected_overlap_with_old':len(set(freeze.selected_candidate_ids)&set(old.selected_candidate_ids)),
            'source_mask_count':int(m.sum()),'frozen_at':utc_now(),'producer_pid':os.getpid(),
            'new_test_read_before_freeze':False,'historical_test_previously_observed':True})
        self.put('selection_freeze.json',freeze.to_dict())
        print('SELECTION_FROZEN',report,flush=True)

    def freeze(self):
        f=SelectionFreeze.from_dict(read_json(self.root/'selection_freeze.json'))
        if f.contract_sha256 != self.sha: raise ValueError('New freeze contract differs')
        return f

    def load_encoding(self, role, wanted=None):
        for path in sorted((self.source/'encoding').glob(role+'-*.json')):
            meta=self.old(str(path.relative_to(self.source)))
            records=meta['records']
            for row in meta['predictions']:
                if (row['oracle_weight_sha256'] != self.c['oracle_sha256']
                        or row['temperature'] != self.c['temperature'] or row['num_classes'] != 3):
                    raise ValueError('Cached original batch32 oracle changed')
                self.pred[row.get('parent_id',row.get('candidate_id'))]=row
            if wanted is not None and not any(r['candidate_id'] in wanted for r in records): continue
            npz=path.with_suffix('.npz')
            if file_sha(npz) != meta['npz_sha256']: raise ValueError('Changed accepted encoding batch')
            with np.load(npz,allow_pickle=False) as z:
                for i,r in enumerate(records):
                    if wanted is None or r['candidate_id'] in wanted:
                        self.enc[r['candidate_id']]={**r,'H':z[f'H{i}'].copy()}

    def exact(self, left, right):
        from src.eval.node_wasserstein_distance import compute_node_wasserstein_distance
        for key in ('producer','molclr_checkpoint_sha256','numerical_contract_sha256',
                    'node_extraction_version','feature_schema_sha256'):
            if left[key] != right[key]: raise ValueError('Raw encoding producer mismatch: '+key)
        if left['molclr_checkpoint_sha256'] != self.c['distance']['molclr_checkpoint_sha256']:
            raise ValueError('Different MolCLR')
        v,_=compute_node_wasserstein_distance(left['H'],right['H'],feature_cost='cosine',node_mass='uniform',size_penalty_beta=0)
        if not np.isfinite(v) or v<0: raise ValueError('Exact WNode failed, not semantic infinity')
        return v

    def evaluate(self):
        self.admission(); f=self.freeze()
        old,ostates,p,oc,mask=self.matrix_old('test'); chosen=list(f.selected_candidate_ids)
        missing=set(chosen)-set(oc)
        if missing:
            self.load_encoding('pool',missing); self.load_encoding('test')
        for block,start in enumerate(range(0,len(p),8)):
            self.admission(); name=f'test/block-{block:04d}'
            if (self.root/(name+'.json')).exists(): continue
            stop=min(start+8,len(p)); d=np.full((stop-start,len(chosen)),np.nan); s=np.zeros(d.shape,dtype=np.uint8)
            computed=reused=0; cache={}; keys=[]
            for i,pid in enumerate(p[start:stop]):
                for j,cid in enumerate(chosen):
                    if cid in oc:
                        d[i,j]=old[start+i,oc.index(cid)];s[i,j]=ostates[start+i,oc.index(cid)];reused+=1;continue
                    if not mask[start+i]: d[i,j]=np.inf;s[i,j]=2;continue
                    before,after=self.pred[pid],self.pred[cid]
                    if before['predicted_label'] != self.c['source_label'] or after['predicted_label'] not in self.c['allowed_destinations']:
                        raise ValueError('Saved source/target mask conflict')
                    left,right=self.enc[before['full_graph_id']],self.enc[after['full_graph_id']]
                    key=tuple(sorted((left['encoding_sha256'],right['encoding_sha256'])))
                    if key not in cache: cache[key]=self.exact(left,right);computed+=1
                    else: reused+=1
                    d[i,j]=cache[key];s[i,j]=1
                    keys.append((pid,cid,left['graph_sha256'],right['graph_sha256'],*key))
            decode_status(d,s);commit_npz(self.root/(name+'.npz'),values=d,states=s)
            self.put(name+'.json',dict(parent_ids=p[start:stop],candidate_ids=chosen,source_mask=mask[start:stop].tolist(),
                contract_sha256=self.sha,freeze_sha256=f.freeze_sha256,npz_sha256=file_sha(self.root/(name+'.npz')),
                computed=computed,reused=reused,raw_pair_identity_sha256=digest(keys),producer_pid=os.getpid()))
            atomic_json(self.root/'progress.json',dict(stage='SELECTED_TEST',completed_blocks=block+1,total_blocks=(len(p)+7)//8,at=utc_now()))
        print('SELECTED_TEST_COMPLETE',len(chosen),'missing columns',len(missing),flush=True)

    def result(self):
        f=self.freeze(); ids=self.c['test_ids']; cols=list(f.selected_candidate_ids)
        d=np.full((len(ids),len(cols)),np.nan);s=np.zeros(d.shape,dtype=np.uint8);mask=[]
        for block,start in enumerate(range(0,len(ids),8)):
            name=f'test/block-{block:04d}';r=read_json(self.root/(name+'.json'));stop=min(start+8,len(ids))
            if (r['parent_ids']!=ids[start:stop] or r['candidate_ids']!=cols or r['contract_sha256']!=self.sha
                    or r['freeze_sha256']!=f.freeze_sha256 or r['npz_sha256']!=file_sha(self.root/(name+'.npz'))):
                raise ValueError('New test block binding differs')
            with np.load(self.root/(name+'.npz'),allow_pickle=False) as z:d[start:stop]=z['values'];s[start:stop]=z['states']
            mask.extend(r['source_mask'])
        return evaluate_frozen_test(f,d,pair_status=decode_status(d,s),parent_ids=ids,candidate_ids=cols,
                                    source_mask=np.asarray(mask,dtype=bool),contract_sha256=self.sha)

    def audit(self):
        self.admission();f=self.freeze();report=read_json(self.root/'selection_report.json')
        if report['producer_pid']==os.getpid():raise ValueError('Separate audit process required')
        d,s,p,c,m=self.matrix_old('calibration')
        replay,_=freeze_k20(d,decode_status(d,s),p,c,m,0.1,self.c['cost_cap'],self.old_spec['evaluation']['grid'],
                            self.sha,self.old('pool_binding.json')['pool_sha256'])
        if replay != f:raise ValueError('Independent selection replay differs')
        result=self.result();metrics=result.prefix_metrics()
        if any(b['covered_count']<a['covered_count'] or b['cost']>a['cost'] for a,b in zip(metrics,metrics[1:])):
            raise ValueError('Nested prefix inconsistency')
        oldcols=set(self.old('test_prepared.json')['candidate_ids']);missing=set(f.selected_candidate_ids)-oldcols
        checks=0
        if missing:
            self.load_encoding('pool',missing);self.load_encoding('test')
            # Audit only new pairs; prior accepted oracle/64-distance audit is adopted.
            for block in range(59):
                name=f'test/block-{block:04d}';r=read_json(self.root/(name+'.json'))
                if r['producer_pid']==os.getpid():raise ValueError('Independent pair producer required')
                with np.load(self.root/(name+'.npz'),allow_pickle=False) as z:
                    for i,pid in enumerate(r['parent_ids']):
                        for j,cid in enumerate(r['candidate_ids']):
                            if cid not in missing or z['states'][i,j]!=1 or checks>=16:continue
                            v=self.exact(self.enc[self.pred[pid]['full_graph_id']],self.enc[self.pred[cid]['full_graph_id']])
                            if v!=z['values'][i,j]:raise ValueError('Independent new raw pair differs')
                            checks+=1
                if checks>=16:break
        self.put('test_evaluation.json',result.to_dict())
        blocks=[read_json(self.root/f'test/block-{i:04d}.json') for i in range(59)]
        self.put('audit/final_audit.json',dict(status='TASTE_V5_CM_ACCEPTED',contract_sha256=self.sha,
            source_acceptance_sha256=digest(self.audit_old),selection_freeze_sha256=f.freeze_sha256,
            new_test_exact_pairs=sum(x['computed'] for x in blocks),reused_test_slots=sum(x['reused'] for x in blocks),
            independent_new_pairs_checked=checks,original_oracle_repeated=False,original_calibration_ot_repeated=False,
            test_evaluation_sha256=digest(result.to_dict()),metrics_k20=metrics[-1],main_authority_written=False))
        print('CM_V5_ACCEPTED',metrics[-1],flush=True)

    def export(self):
        a=read_json(self.root/'audit/final_audit.json')
        if a['status']!='TASTE_V5_CM_ACCEPTED' or a['contract_sha256']!=self.sha:raise ValueError('No acceptance')
        result=self.result()
        write_csv(self.root/'source_csv/prefix_metrics.csv',result.prefix_metrics())
        write_csv(self.root/'source_csv/parent_best_distances.csv',result.parent_best_rows())
        write_csv(self.root/'source_csv/exact_ecdf_k20.csv',result.exact_ecdf(20))


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--contract',required=True)
    p.add_argument('--source-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--action',required=True,choices=['select','evaluate','audit','export','status'])
    a=p.parse_args(argv);x=TasteCM(a.contract,a.source_root,a.output_root)
    if a.action=='status':
        for name in ['progress.json','selection_report.json','audit/final_audit.json']:
            if (x.root/name).exists():print(name,read_json(x.root/name))
    else:getattr(x,a.action)()
