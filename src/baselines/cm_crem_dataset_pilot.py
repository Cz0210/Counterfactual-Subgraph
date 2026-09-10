"""Shared train-only CM pilot using original RF or Taste three-class GINE.

Reuses original attribution, native generation, structural sampling and exact
WNode. This module owns bounded pilot work, not a new scheduler or authority.
"""
from __future__ import annotations
import csv, json, os, resource, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .cm_crem_runtime import atomic_json, read_json, file_sha, digest, require_compute_node

TERMINALS={'GENERATED','GENERATION_COMPLETE','NO_NATIVE_REPLACEMENT','NO_REPLACEABLE_CONTEXT','TIMEOUT_BUDGETED'}


def serial_rf_execution(oracle):
    """Fix only tree-reduction scheduling, never the forest or feature vector.

    Job2667864 reproduced unequal repeated predictions at n_jobs=7 and exact
    three-repeat attribution at n_jobs=1. Keep strict comparisons, not tolerance.
    """
    if not hasattr(oracle.model,'estimators_') or not hasattr(oracle.model,'n_jobs'):
        raise ValueError('Unsupported RF execution interface')
    previous=oracle.model.n_jobs
    oracle.model.n_jobs=1
    oracle.cm_execution_receipt={'method':'SERIAL_TREE_REDUCTION','original_n_jobs':previous,
                                 'effective_n_jobs':1,'weights_changed':False,
                                 'feature_schema_changed':False,'tolerance_changed':False}
    return oracle


def validate(spec):
    expected={'seed':7,'mask_fraction':0.2,'radius':1,'min_max_inc':3,
              'max_replacements_per_component':64,'raw_max_per_parent':128,
              'parent_wall_seconds':900,'parent_limit':32,'full_library_cap':6000}
    for key,value in expected.items():
        if spec.get(key)!=value:raise ValueError('Frozen CM parameter mismatch: '+key)
    if spec['dataset'] not in {'tastemolnet','mutagenicity','aids'}:raise ValueError('CM dataset outside scope')
    classes=3 if spec['dataset']=='tastemolnet' else 2
    if spec['num_classes']!=classes or spec['source_label']!=1 or spec['allowed_destinations']!=([0,2] if classes==3 else [0]):
        raise ValueError('Original source/target mapping differs')
    if (spec['dataset']=='tastemolnet') != (spec['oracle_backend']=='gine'):raise ValueError('Original oracle family differs')
    if spec['deadline_utc']!='2026-09-16T16:27:44Z':raise ValueError('Original deadline reset')


def oracle_for(spec):
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer, MolecularFeatureSchema
    if spec['oracle_backend']=='rf':
        from src.oracles.oracle_factory import LegacyRFOracle
        if file_sha(spec['oracle_path'])!=spec['oracle_sha256']:raise ValueError('Actual dataset RF identity differs')
        oracle=LegacyRFOracle(spec['oracle_path'],num_classes=2,source_label=1)
        if tuple(oracle.class_labels)!=(0,1):raise ValueError('RF class axes need explicit mapping')
        serial_rf_execution(oracle)
        return oracle, MolecularGraphFeaturizer()
    from src.oracles.gnn_oracle import GNNOracle
    from .cm_crem_oracle import FrozenCMOracle
    root=Path(spec['oracle_path'])
    for name,sha in spec['oracle_files'].items():
        if file_sha(root/name)!=sha:raise ValueError('Taste original bundle differs: '+name)
    card=read_json(root/'model_card.json');temp=read_json(root/'temperature_scaling.json')
    if str(card['dataset']).lower()!='tastemolnet' or card['backbone']!='gine':raise ValueError('Not original Taste GINE')
    if temp['status']!='fit' or temp.get('fit_split',temp.get('selection_split')) not in {'val','validation'}:
        raise ValueError('Original validation temperature fit missing')
    gnn=GNNOracle.from_checkpoint(root,device='cpu',batch_size=32)
    if gnn.checkpoint_id!=spec['oracle_sha256']:raise ValueError('Taste weight identity differs')
    feature=MolecularGraphFeaturizer(MolecularFeatureSchema.from_dict(read_json(root/'feature_schema.json')))
    return FrozenCMOracle(gnn,feature,binding={'dataset':'tastemolnet','allowed_destinations':[0,2],
                          'temperature_sha256':spec['oracle_files']['temperature_scaling.json']}), feature


def read_train(spec):
    path=Path(spec['train']['path'])
    if file_sha(path)!=spec['train']['sha256']:raise ValueError('Train manifest content differs')
    rows=list(csv.DictReader(path.open()));binding=spec['train'];out=[]
    for i,row in enumerate(rows):
        if int(row[binding['label_field']])!=spec['source_label']:continue
        identity=str(row[binding['id_field']]) if binding.get('id_field') else spec['dataset']+'_CM_train_'+digest([i,row[binding['smiles_field']]])[:16]
        out.append({'parent_id':identity,'smiles':row[binding['smiles_field']],'split':'train','label':1,'source_row':i})
    if len({r['parent_id'] for r in out})!=len(out):raise ValueError('Duplicate authoritative train IDs')
    return out,len(rows)


class DatasetPilot:
    def __init__(self,path):
        self.path=Path(path);self.spec=read_json(path);validate(self.spec)
        self.root=Path(self.spec['output_root'])
        if not self.root.is_relative_to('/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2'):
            raise ValueError('HPC project scope')
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parents[2],text=True).strip()
        if actual!=self.spec['execution_commit']:raise ValueError('Immutable producer commit differs')
        self.sha=digest({k:v for k,v in self.spec.items() if k not in {'execution_commit','execution_root','output_root'}})
    def put(self,name,value):
        return atomic_json(self.root/name,{'scope_sha256':self.sha,**value},immutable=True)
    def get(self,name):
        v=read_json(self.root/name)
        if v.get('scope_sha256')!=self.sha:raise ValueError('Pilot mixed scope: '+name)
        return v
    def deadline(self):
        if datetime.now(timezone.utc)>=datetime.fromisoformat(self.spec['deadline_utc'].replace('Z','+00:00')):
            raise RuntimeError('Original deadline reached; no new parent started')
    def attribute(self):
        if (self.root/'attribution.json').exists():return self.get('attribution.json')
        from .cm_crem_experiment import pilot_indices
        oracle,_=oracle_for(self.spec);parents,total=read_train(self.spec)
        if self.spec['oracle_backend']=='rf':
            self.put('rf_execution_receipt.json',oracle.cm_execution_receipt)
        if self.spec['oracle_backend']=='rf':
            preds=oracle.predict_proba([p['smiles'] for p in parents]);labels=[int(x.argmax()) for x in preds]
        else:
            preds=oracle.predict_rows(parents,split='train');labels=[p['predicted_label'] for p in preds]
        eligible=[p for p,label in zip(parents,labels,strict=True) if label==1]
        self.put('train_source_manifest.json',{'parents':eligible,'all_train_count':total,'label_source_count':len(parents),
                  'predicted_source_count':len(eligible),'test_read':False,'train_file_sha256':self.spec['train']['sha256']})
        indices,structure=pilot_indices(eligible,self.sha);selected=[eligible[i] for i in indices]
        self.put('pilot_manifest.json',{'parents':selected,'structure':structure,'selected_before_generation':True,'outcome_used_for_sampling':False})
        records=[]
        for parent in selected:
            name='attribution_units/'+digest(parent['parent_id'])[:20]+'.json'
            if (self.root/name).exists():a=self.get(name)['record']
            else:
                self.deadline()
                if self.spec['oracle_backend']=='rf':
                    from .cm_crem_rf_attribution import attribute
                    a=attribute(oracle,parent)
                    if a!=attribute(oracle,parent):raise ValueError('RF attribution repeat differs')
                else:a=oracle.attribute_train_parent(parent)
                self.put(name,{'record':a})
            records.append(a)
        return self.put('attribution.json',{'records':records,'parents':selected,'count':32,'test_read':False})
    def generate(self):
        from .cm_crem_assets import prepare_job_scratch,stage_static_database,validate_database_source
        from .cm_crem_generation import generate_parent
        e=self.spec['generation_execution'];receipt=read_json(e['database_receipt'])
        if Path(sys.executable).resolve()!=Path(e['generator_python']).resolve():raise ValueError('Wrong CReM environment')
        old=read_json(self.spec['original_cm_spec']);source=validate_database_source(old,receipt)
        scratch=prepare_job_scratch(self.root,required_bytes=receipt['uncompressed_bytes'],reserve_bytes=2*1024**3)
        self.put('scratch-'+os.environ['SLURM_JOB_ID']+'.json',scratch)
        if scratch['status']!='JOB_SCRATCH_READY':raise RuntimeError(scratch)
        staged=stage_static_database(e['database_path'],e['database_receipt'],reserve_bytes=2*1024**3,expected_url=source['actual_source_url'],scratch_receipt=scratch)
        if staged['status']!='LOCAL_DATABASE_READY':raise RuntimeError(staged)
        self.put('database-'+os.environ['SLURM_JOB_ID']+'.json',staged)
        for a in self.get('attribution.json')['records']:
            name='generated/'+digest(a['parent_id'])[:20]+'.json'
            if (self.root/name).exists():
                value=self.get(name)
            else:
                self.deadline()
                value=generate_parent(a['generation_request'],{'database_path':staged['database_path'],'upstream_root':e['upstream_root'],
                    'science_hash':self.sha,'parent_wall_limit_seconds':900},log_path=self.root/'logs'/('native-'+a['parent_id']+'.log'))
                self.put(name,value)
            if value['status'] not in TERMINALS:raise RuntimeError('Native generation failed: '+str(value))
        self.put('generation_complete.json',{'count':32,'generation_budget_unchanged':True})
    def verify(self):
        from rdkit import Chem
        from .cm_crem_oracle import graph_identity,FrozenCMWNode,raw_distance_record
        oracle,feature=oracle_for(self.spec);attrs=self.get('attribution.json');rows=[];qualified={};times=[]
        start=time.monotonic()
        for p,a in zip(attrs['parents'],attrs['records'],strict=True):
            g=self.get('generated/'+digest(p['parent_id'])[:20]+'.json')
            if g['status'] not in TERMINALS:raise ValueError('Failed generation is not zero')
            if self.spec['oracle_backend']=='gine':f=oracle.filter_generated(p,g)
            else:
                valid=[];rejected=[]
                for r in g['retained_raw']:
                    try:identity=graph_identity(r['smiles'],feature)
                    except (ValueError,RuntimeError) as e:rejected.append({'raw_id':r['raw_id'],'reason':str(e)});continue
                    probs=oracle.predict_proba([r['smiles']])[0];label=int(probs.argmax())
                    c={**identity,'smiles':identity['canonical_smiles'],'probabilities':probs.tolist(),'predicted_label':label,
                       'origins':[{'parent_id':p['parent_id'],'raw_id':r['raw_id']}],'oracle_weight_sha256':self.spec['oracle_sha256']}
                    if label in self.spec['allowed_destinations']:valid.append(c)
                    else:rejected.append({'raw_id':r['raw_id'],'reason':'NOT_DESTINATION'})
                f={'parent_id':p['parent_id'],'accepted':valid,'rejected':rejected,'raw_count':len(g['retained_raw'])}
            self.put('filter/'+digest(p['parent_id'])[:20]+'.json',f);rows.append({'parent_id':p['parent_id'],'raw_count':len(g['retained_raw']),'accepted':len(f['accepted']),'status':g['status']})
            qualified.update({c['candidate_id']:c for c in f['accepted']});times.append(g.get('parent_wall_seconds',g.get('elapsed_seconds',0)))
        self.put('pilot_pool.json',{'candidates':list(qualified.values()),'rows':rows,'test_read':False})
        distance=FrozenCMWNode.from_resolved(self.spec['resolved_wnode']);enc=distance.encode_rows(attrs['parents'],featurizer=feature);pairs=[]
        # Use all 32 parents twice, not only the smallest molecule's neighbours.
        for delta in (1,7):
            for i,left in enumerate(enc):
                right=enc[(i+delta)%len(enc)]
                if left['candidate_id']==right['candidate_id']:raise ValueError('Pilot timing self pair requires explicit alternate')
                t=time.monotonic();raw=raw_distance_record(left,right,numerical_contract=self.spec['resolved_wnode'])
                pairs.append({'raw':raw,'seconds':time.monotonic()-t})
        self.put('wnode_timing.json',{'pairs':pairs,'count':64,'oracle_and_distance_scope':self.spec['dataset'],'test_read':False})
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        return self.put('pilot_receipt.json',{'status':'PILOT32_GENERATION_FILTER_WNODE_COMPLETE','parent_count':32,'unique_target_count':len(qualified),
            'generation_seconds':times,'generation_p90_seconds':float(np.quantile(times,.9)),'wnode_p90_seconds':float(np.quantile([p['seconds'] for p in pairs],.9)),
            'filter_and_encoding_seconds':time.monotonic()-start,'process_peak_rss_bytes':peak,'new_training':False,'test_read':False,
            'full_science_complete':False,'full_dispatch_requires':'complete dataset calibration/test binding and measured full-stage admission'})
    def run(self):
        self.attribute()
        env=dict(os.environ,PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
        subprocess.run([self.spec['generation_execution']['generator_python'],'-s','-B',str(Path(__file__).parents[2]/'scripts/run_cm_crem_dataset_pilot.py'),
                       '--config','configs/hpc.yaml','--spec',str(self.path),'--action','generate'],env=env,check=True)
        return self.verify()


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--action',choices=['run','attribute','generate','verify','status'],required=True);a=p.parse_args(argv)
    x=DatasetPilot(a.spec)
    if a.action=='status':
        print(json.dumps({'root':str(x.root),'pilot_receipt':read_json(x.root/'pilot_receipt.json') if (x.root/'pilot_receipt.json').exists() else None}));return
    require_compute_node();x.deadline();print(json.dumps(getattr(x,a.action)()))
