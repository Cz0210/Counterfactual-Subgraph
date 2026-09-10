"""BACE CM saved5474-pool continuation, one bounded Slurm execution chain."""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .cm_crem_runtime import atomic_json,read_json,file_sha,digest,require_compute_node,checked_root,utc_now
from .cm_crem_experiment import Experiment,load_parent_rows
from .cm_crem_k20 import freeze_k20,objective
from .cm_crem_selection import SelectionFreeze,evaluate_frozen_test


class K20Experiment:
    def __init__(self, config):
        self.config=read_json(config);self.config_path=Path(config)
        self.root=checked_root(self.config['output_root'],'/share/home/u20526/czx')
        self.old=Experiment(Path(self.config['source_spec']),Path(self.config['source_root']))
        self.spec=self.old.spec;self.root.mkdir(parents=True,exist_ok=True)
        code=Path(__file__).parents[2]
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
        if actual!=self.config['execution_commit'] or str(code)!=self.config['execution_root']:
            raise ValueError('Immutable K20 execution binding mismatch')
        if self.config['deadline_utc']!='2026-09-16T16:27:44Z':raise ValueError('Original deadline changed')
        if self.config['max_swap_rounds']!=2 or self.config['generation_enabled'] is not False:
            raise ValueError('Authorized fixed-generation K20 scope changed')
        audit=self.old.get('audit/final_audit.json')
        if audit['status']!='BACE_CM_CREM_FINAL_AUDIT_PASS':raise ValueError('Original scientific acceptance missing')
        self.pins=self.old.get('audit/provenance_review.json')['bound_record_sha256']
        if self.config['source_science_sha']!=self.old.sha:raise ValueError('Wrong original experiment')
        self.sha=digest({k:v for k,v in self.config.items() if k not in {'execution_root','execution_commit','output_root','source_root','source_spec'}})

    def deadline(self):
        if datetime.now(timezone.utc)>=datetime.fromisoformat(self.config['deadline_utc'].replace('Z','+00:00')):
            raise RuntimeError('Original deadline reached; completed units retained')

    def oldread(self,name):
        if name not in self.pins or file_sha(self.old.root/name)!=self.pins[name]:
            raise ValueError('Original producer binding missing/changed: '+name)
        return self.old.get(name)

    def put(self,name,data):
        out={'scope_sha256':self.sha,**data};atomic_json(self.root/name,out,immutable=True);return out

    def get(self,name):
        out=read_json(self.root/name)
        if out.get('scope_sha256')!=self.sha:raise ValueError('K20 scope changed: '+name)
        return out

    def prepare(self):
        if (self.root/'pool.json').exists():return self.get('pool.json')
        attrs=self.oldread('attribution.json');accepted={};unitpins={}
        for parent in attrs['parents']:
            name='filter_units/'+digest(parent['parent_id'])[:20]+'.json';record=self.oldread(name)
            unitpins[name]=self.pins[name]
            if record['status']!='FILTER_COMPLETE' or record['source_prediction']['predicted_label']!=1:
                raise ValueError('Unqualified train source')
            for c in record['accepted']:
                if c['predicted_label']!=0 or c['oracle_weight_sha256']!=self.spec['resolved_oracle']['model_sha256']:
                    raise ValueError('Candidate oracle qualification changed')
                accepted.setdefault(c['candidate_id'],c)
        oldpool=self.oldread('pool_freeze.json');oldids=oldpool['candidate_ids']
        ids=sorted(accepted,key=lambda c:(digest([self.old.sha,7,c]),c))
        if len(ids)!=5474 or ids[:2000]!=oldids:raise ValueError('Actual full pool/count/P2000 subset conflict')
        return self.put('pool.json',{'status':'TRAIN_FIXED_POOL_5474','ids':ids,'candidates':[accepted[x] for x in ids],
            'p2000':oldids,'source_filter_pins':unitpins,'generation_calls':0,'pool_selected_with_calibration':False,
            'pool_selected_with_test':False,'post_hoc_test_previously_observed':True})

    def encodings(self):
        pool=self.get('pool.json');old=self.oldread('pool_encodings.json')
        enc={r['candidate_id']:r for r in old['records']}
        missing=[c for c in pool['candidates'] if c['candidate_id'] not in enc]
        oracle=distance=None
        for start in range(0,len(missing),32):
            self.deadline();name=f'encodings/block-{start//32:04d}.json'
            if (self.root/name).exists():rows=self.get(name)['records']
            else:
                if oracle is None:oracle=self.old.oracle();distance=self.old.distance()
                rows=distance.encode_rows(missing[start:start+32],featurizer=oracle.featurizer)
                self.put(name,{'records':rows,'original_molclr_contract':self.spec['resolved_wnode'],
                    'new_graphs_only':True,'job_id':os.environ['SLURM_JOB_ID']})
            if [r['candidate_id'] for r in rows]!=[r['candidate_id'] for r in missing[start:start+32]]:
                raise ValueError('Encoding block omitted/reordered graphs')
            enc.update({r['candidate_id']:r for r in rows})
        if set(enc)!=set(pool['ids']):raise ValueError('Incomplete full pool encodings')
        return enc

    def matrix_stage(self,split):
        from .cm_crem_oracle import raw_distance_record,full_graph_pair
        from .cm_crem_audit import validate_raw_pair
        if (self.root/(split+'/complete.json')).exists():return self.get(split+'/complete.json')
        pool=self.get('pool.json');byid={c['candidate_id']:c for c in pool['candidates']}
        if split=='test':
            freeze=SelectionFreeze.from_dict(self.get('freeze.json')['freeze'])
            ids=list(freeze.selected_candidate_ids)
        else:ids=pool['ids']
        # Original base manifests, never v2 test-wide pool evaluation.
        parents=load_parent_rows(self.spec,split,self.old.root)
        enc=self.encodings();distance=oracle=None;reused=calculated=0
        for parent in parents:
            self.deadline();name=f'{split}/{digest(parent["parent_id"])[:20]}.json'
            if (self.root/name).exists():continue
            started=time.monotonic()
            oldname=f'{split}/parents/{digest(parent["parent_id"])[:20]}.json';old=self.oldread(oldname)
            prediction=old['prediction'];left=old['parent_encoding'];cache={r['candidate_id']:r for r in old['pairs']}
            if prediction['parent_id']!=parent['parent_id'] or prediction['smiles']!=parent['smiles']:
                raise ValueError('Base parent identity conflict')
            source=prediction['predicted_label']==1
            if source and left is None:raise ValueError('Missing original parent encoding')
            pairs=[]
            for cid in ids:
                prototype=byid[cid]
                if cid in cache:
                    pair=cache[cid]
                    if source:validate_raw_pair(pair['raw_distance'],left,enc[cid],self.spec['resolved_wnode'])
                    reused+=int(source)
                else:
                    raw=raw_distance_record(left,enc[cid],numerical_contract=self.spec['resolved_wnode']) if source else None
                    pair={**full_graph_pair(prediction,prototype,raw_distance=raw),'raw_distance':raw,
                          'pair_status':'OK' if source else 'BEFORE_NOT_SOURCE'}
                    calculated+=int(source)
                pairs.append(pair)
            self.put(name,{'parent_id':parent['parent_id'],'prediction':prediction,'parent_encoding':left,
                'candidate_ids':ids,'pairs':pairs,'old_parent_record_sha':self.pins[oldname],
                'job_id':os.environ['SLURM_JOB_ID'],'test_pool_wide_computed':False})
            atomic_json(self.root/'progress.json',{'stage':split,'completed_parent':parent['parent_id'],
                'new_pairs_this_attempt':calculated,'reused_pairs_this_attempt':reused,'updated_at':utc_now()})
            atomic_json(self.root/'latest_timing.json',{'split':split,'parent_id':parent['parent_id'],
                'seconds':time.monotonic()-started,'candidate_pairs':len(ids),'job_id':os.environ['SLURM_JOB_ID']})
        return self.put(split+'/complete.json',{'parents':len(parents),'candidate_count':len(ids),
            'test_reads_after_freeze':split=='test'})

    def matrix(self,split):
        parents=load_parent_rows(self.spec,split,self.old.root)
        ids=self.get('pool.json')['ids'] if split=='calibration' else self.get('freeze.json')['freeze']['selected_candidate_ids']
        values=[];statuses=[];mask=[]
        for parent in parents:
            r=self.get(f'{split}/{digest(parent["parent_id"])[:20]}.json')
            if r['candidate_ids']!=ids:raise ValueError('Missing or reordered matrix columns')
            values.append([np.inf if p['distance'] is None else p['distance'] for p in r['pairs']])
            statuses.append([p['pair_status'] for p in r['pairs']]);mask.append(r['prediction']['predicted_label']==1)
        return np.array(values,dtype=np.float64),np.array(statuses),[p['parent_id'] for p in parents],ids,np.array(mask,dtype=bool)

    def select(self):
        if (self.root/'freeze.json').exists():return self.get('freeze.json')
        d,st,pids,ids,mask=self.matrix('calibration');ev=self.spec['resolved_evaluation'];old=self.oldread('selection_freeze.json')
        original=list(SelectionFreeze.from_dict(old).selected_candidate_ids);poolsha=digest(self.get('pool.json'))
        args=(ev['theta'],ev['cap'],ev['threshold_grid'],self.sha,poolsha,original)
        small,rs=freeze_k20(d[:,:2000],st[:,:2000],pids,ids[:2000],mask,*args)
        full,rf=freeze_k20(d,st,pids,ids,mask,*args)
        choices=[(small,rs),(full,rf)]
        chosen,report=min(choices,key=lambda x:tuple(-a for a in x[1]['objective'])+(tuple(sorted(x[0].selected_candidate_ids)),))
        oldscore=objective(d[:,[ids.index(x) for x in original]].min(axis=1),ev['theta'],ev['cap'],np.array(ev['threshold_grid']))
        quantiles={label:np.quantile(m[np.isfinite(m)],[0,.25,.5,.75,.9,1]).tolist()
                   if np.isfinite(m).any() else None for label,m in [('p2000',d[:,:2000]),('full5474',d)]}
        self.put('calibration_comparison.json',{'old_v1':oldscore,'p2000_v2':rs,'full5474_v2':rf,
            'finite_pair_quantiles':quantiles,'finite_quantile_levels':[0,.25,.5,.75,.9,1],
            'new_low_threshold_parents':[pids[i] for i in range(len(pids)) if d[i,2000:].min()<=ev['theta']<d[i,:2000].min()],
            'calibration_only':True,'test_used_for_choice':False,'localmask_generation_performed':False,
            'localmask_gate':'NO_FULL_BRANCH_TIMING_PROOF_LOCALMASK_NOT_DISPATCHED'})
        return self.put('freeze.json',{'freeze':chosen.to_dict(),'chosen_calibration_objective':report['objective'],
            'frozen_at':utc_now(),'test_loaded_by_this_run':False,'candidate_budget_comparisons':3})

    def evaluate(self):
        d,st,pids,ids,mask=self.matrix('test');f=self.get('freeze.json')['freeze']
        result=evaluate_frozen_test(f,d,pair_status=st,parent_ids=pids,candidate_ids=ids,source_mask=mask,contract_sha256=self.sha)
        return self.put('test_evaluation.json',result.to_dict())

    def audit(self):
        from .cm_crem_oracle import encoding_digest,raw_distance_record
        from .cm_crem_audit import validate_raw_pair,original_filter_context
        # Independent process, outcome-independent IDs; no v1 audit replay.
        enc=self.encodings();pool=self.get('pool.json');candidates={c['candidate_id']:c for c in pool['candidates']}
        freeze=SelectionFreeze.from_dict(self.get('freeze.json')['freeze'])
        oracle=self.old.oracle();distance=self.old.distance();checked=[]
        for cid in sorted(freeze.selected_candidate_ids)[:2]:
            c=candidates[cid];ctx,pos,origin=original_filter_context(oracle,self.oldread,c)
            fresh=oracle.predict_rows(ctx,split='train_generated')[pos]
            if fresh['logits']!=c['logits'] or fresh['probabilities']!=c['probabilities']:
                raise ValueError('Independent original batch prototype prediction differs')
            re=distance.encode_rows([c],featurizer=oracle.featurizer)[0]
            if re['encoding_sha256']!=enc[cid]['encoding_sha256']:raise ValueError('New prototype encoding not reproducible')
            checked.append(cid)
        ot_checks=[]
        for split in ('calibration','test'):
            for parent in load_parent_rows(self.spec,split,self.old.root):
                row=self.get(f'{split}/{digest(parent["parent_id"])[:20]}.json')
                for pair in row['pairs']:
                    if pair['raw_distance'] is not None:
                        validate_raw_pair(pair['raw_distance'],row['parent_encoding'],enc[pair['candidate_id']],self.spec['resolved_wnode'])
                        if len(ot_checks)<2 and pair['candidate_id'] not in pool['p2000']:
                            fresh=raw_distance_record(row['parent_encoding'],enc[pair['candidate_id']],numerical_contract=self.spec['resolved_wnode'])
                            if fresh!=pair['raw_distance']:raise ValueError('Independent new raw distance differs')
                            ot_checks.append([parent['parent_id'],pair['candidate_id']])
        # Independent calibration-only selector replay; no test values enter it.
        d,st,pids,ids,mask=self.matrix('calibration');ev=self.spec['resolved_evaluation']
        original=list(SelectionFreeze.from_dict(self.oldread('selection_freeze.json')).selected_candidate_ids)
        args=(ev['theta'],ev['cap'],ev['threshold_grid'],self.sha,digest(pool),original)
        small,rs=freeze_k20(d[:,:2000],st[:,:2000],pids,ids[:2000],mask,*args)
        full,rf=freeze_k20(d,st,pids,ids,mask,*args)
        replay,_=min([(small,rs),(full,rf)],key=lambda x:tuple(-a for a in x[1]['objective'])+(tuple(sorted(x[0].selected_candidate_ids)),))
        if replay.freeze_sha256!=freeze.freeze_sha256:raise ValueError('Independent calibration selection differs')
        # Re-evaluate frozen prefixes, not select from test.
        before=self.get('test_evaluation.json');self.evaluate()
        if before!=self.get('test_evaluation.json'):raise ValueError('Independent prefix reconciliation differs')
        return self.put('audit/k20_audit.json',{'status':'K20_RECORDS_AND_INDEPENDENT_PROTOTYPE_CHECKS_PASS',
            'checked_prototypes':checked,'selection_freeze_sha':freeze.freeze_sha256,'test_count':141,
            'independent_new_ot_checks':ot_checks,'independent_calibration_replay':True,
            'original_generation_reused':True,'original_audit_not_rerun':True,'new_generation_calls':0,
            'scope':'BACE_ORIGINAL_GINE_FULL_TRAIN_POOL_K20_POSTHOC','main_matrix_written':False})

    def export(self):
        from .cm_crem_export import export_results
        audit=self.get('audit/k20_audit.json')
        if audit['status']!='K20_RECORDS_AND_INDEPENDENT_PROTOTYPE_CHECKS_PASS':raise ValueError('K20 audit missing')
        raw=self.get('test_evaluation.json').copy();raw.pop('scope_sha256')
        return export_results(raw,self.root,dataset='bace',oracle='gine',k20_reuse_audit=audit)


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/hpc.yaml');p.add_argument('--spec',required=True)
    p.add_argument('--action',choices=['run','prepare','calibrate','select','test','audit','export','status'],required=True)
    a=p.parse_args(argv);x=K20Experiment(a.spec)
    if a.action=='status':
        print(json.dumps({'root':str(x.root),'progress':read_json(x.root/'progress.json') if (x.root/'progress.json').exists() else None}));return
    require_compute_node();x.deadline()
    if a.action=='run':
        x.prepare();x.matrix_stage('calibration');x.select();x.matrix_stage('test');x.evaluate()
        subprocess.run([sys.executable,'-I','-B',str(Path(__file__).parents[2]/'scripts/run_cm_crem_k20.py'),
            '--config',a.config,'--spec',a.spec,'--action','audit'],check=True)
        x.export()
    elif a.action=='calibrate':x.matrix_stage('calibration')
    elif a.action=='test':x.matrix_stage('test');x.evaluate()
    else:getattr(x,a.action)()
    print(json.dumps({'action':a.action,'completed':True,'root':str(x.root)}))
